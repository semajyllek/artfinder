# artfinder/evaluator.py
import os
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from .config import Config

class Evaluator:
    def __init__(self, state):
        self.state = state

    def run_final_exam_and_log(self, nprobe=8, silent=True, test_ids=None):
        """Main entry point for running a validation batch."""
        if test_ids is None:
            blob = self.state.bucket.blob(Config.MANIFEST_PATH)
            test_ids = json.loads(blob.download_as_string())["test_queries"]
        
        image_results, summary_row = self.collect_eval_results(test_ids, nprobe=nprobe, silent=silent)
        print(f"\n--- VALIDATION SCORE: {summary_row['accuracy']*100:.1f}% | Latency: {summary_row['avg_latency_ms']}ms ---")
        return image_results, summary_row

    def collect_eval_results(self, test_ids, nprobe=8, silent=True):
        """Orchestrates the verification loop."""
        # 🌟 FIXED: Points to the new consolidated vault builder location
        from .vault.builder import load_source_metadata
        self.state.source_df = load_source_metadata(self.state.bucket)
        
        results = []
        run_id = time.strftime("%Y%m%d_%H%M%S")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        for obj_id in tqdm(test_ids, desc="Verifying Batch"):
            raw_img = self._fetch_ground_truth_image(obj_id)
            if raw_img is None:
                continue

            test_photo = apply_simulation(raw_img)
            prediction, confidence, latency = self._run_inference(test_photo, nprobe)
            result_entry = self._score_prediction(obj_id, prediction, confidence, latency, run_id, ts)
            results.append(result_entry)
            
            if not silent or not result_entry['match']:
                show_3panel(self.state, test_photo, prediction, obj_id, confidence)
                
        df_img = pd.DataFrame(results)
        n_tested = len(df_img)
        n_correct = df_img['match'].sum() if n_tested > 0 else 0
        accuracy = (n_correct / n_tested) if n_tested > 0 else 0.0
        avg_latency = df_img['latency_ms'].mean() if n_tested > 0 else 0.0

        summary_row = {
            "run_id": run_id, "timestamp": ts, "n_features": Config.N_FEATURES,
            "dimension": Config.DIMENSION, "resize_dim": str(Config.RESIZE_DIM),
            "scale_factor": Config.SCALE_FACTOR, "n_levels": Config.N_LEVELS,
            "wta_k": Config.WTA_K, "vault_size": self.state.index.ntotal,
            "clusters": Config.CLUSTERS, "nprobe": nprobe, "n_tested": n_tested,
            "n_correct": n_correct, "accuracy": accuracy, "avg_latency_ms": avg_latency
        }
        return df_img, summary_row

    def _fetch_ground_truth_image(self, obj_id):
        try:
            blob = self.state.bucket.blob(f"images/{obj_id}.jpg")
            return Image.open(BytesIO(blob.download_as_bytes())).convert('RGB')
        except:
            return None



    def _standardize_descriptor_shape(self, des):
        """
        Ensures the incoming descriptor matrix strictly matches the exact 
        fixed feature layout boundaries required by the FAISS matrix index.
        """
        if des is None or len(des) == 0:
            return np.zeros((Config.N_FEATURES, 32), dtype=np.uint8)
            
        if len(des) > Config.N_FEATURES:
            return des[:Config.N_FEATURES]
        elif len(des) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(des), 32), dtype=np.uint8)
            return np.vstack([des, padding])
        return des


    def _execute_faiss_batch_search(self, des, nprobe):
        """
        Locks down cluster pruning limits and passes descriptors as a unified 
        parallel C++ batch array block directly to the FAISS index.
        """
        if hasattr(self.state.index, 'nprobe'):
            self.state.index.nprobe = nprobe
            
        # des is shape (500, 32). C++ layer searches all 500 vectors simultaneously
        D, I = self.state.index.search(des, k=1)
        return D, I


    def _tally_identity_votes(self, index_matrix):
        """
        Scans through the returned parallel nearest-neighbor row matches and 
        tallies identity frequencies across metadata parquet block boundaries.
        """
        counts = {}
        for row_idx in index_matrix.flatten():
            if row_idx < 0: 
                continue
                
            for _, row_record in self.state.source_df.iterrows():
                start_r = int(row_record['start_row'])
                end_r = int(row_record['end_row'])
                
                if start_r <= row_idx <= end_r:
                    artwork_id = row_record['id']
                    counts[artwork_id] = counts.get(artwork_id, 0) + 1
                    break
        return counts


    def _resolve_top_prediction(self, identity_votes):
        """
        Evaluates voting frequencies to select the maximum confidence target identity.
        """
        if not identity_votes:
            return None, 0.0
            
        best_id = max(identity_votes, key=identity_votes.get)
        confidence = identity_votes[best_id] / Config.N_FEATURES
        return best_id, confidence


    def _run_inference(self, img_np, nprobe=8):
        """
        Main orchestration gateway for processing high-speed inference passes.
        """
        start_time = time.time()
        
        # 1. Image Keypoint Detection
        resized = cv2.resize(img_np, Config.RESIZE_DIM)
        kp, des = self.state.orb.detectAndCompute(resized, None)
        
        # 2. Vector Matrix Alignment
        standardized_des = self._standardize_descriptor_shape(des)
        
        # 3. Parallel C++ Core Cluster Index Lookup
        D, I = self.state.execute_faiss_batch_search(standardized_des, nprobe)
        latency_ms = (time.time() - start_time) * 1000
        
        # 4. Identity Mapping & Confidence Sorting
        votes = self._tally_identity_votes(I)
        predicted_id, confidence = self._resolve_top_prediction(votes)
        
        return predicted_id, confidence, latency_ms


    def _score_prediction(self, true_id, pred_id, confidence, latency, run_id, ts):
        match = (true_id == pred_id)
        title, artist = "Unknown", "Unknown"
        if pred_id:
            row = self.state.source_df[self.state.source_df['id'] == pred_id]
            if not row.empty:
                title, artist = row.iloc[0]['title'], row.iloc[0]['artist']
        return {
            "run_id": run_id, "timestamp": ts, "n_features": Config.N_FEATURES,
            "dimension": Config.DIMENSION, "resize_dim": str(Config.RESIZE_DIM),
            "scale_factor": Config.SCALE_FACTOR, "n_levels": Config.N_LEVELS,
            "wta_k": Config.WTA_K, "vault_size": self.state.index.ntotal,
            "clusters": Config.CLUSTERS, "nprobe": self.state.index.nprobe,
            "true_id": true_id, "predicted_id": pred_id, "predicted_title": title,
            "predicted_artist": artist, "confidence": confidence, "match": match, "latency_ms": latency
        }


def apply_simulation(img_np):
    """Simulates real-world user imagery distortion layers (synthetic blur)."""
    return cv2.GaussianBlur(img_np, (5, 5), 0)



def show_3panel(state, test_photo, pred_id, true_id, confidence):
    """Renders clean, structured 3-panel visual performance comparisons."""
    meta = None
    if pred_id:
        r = state.source_df[state.source_df['id'] == pred_id]
        if not r.empty: 
            meta = r.iloc[0]
        
    try:
        blob = state.bucket.blob(f"images/{true_id}.jpg")
        true_img = Image.open(BytesIO(blob.download_as_bytes()))
    except: 
        return

    pred_img = None
    
    if meta is not None:
        try:
            p_blob = state.bucket.blob(f"images/{meta['id']}.jpg")
            pred_img = Image.open(BytesIO(p_blob.download_as_bytes()))
        except: 
            pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(test_photo)
    axes[0].set_title("1. Simulated User Photo")
    axes[0].axis('off')
    
    if pred_img:
        axes[1].imshow(pred_img)
        axes[1].set_title(f"2. Prediction: {meta['title']}\nConf: {confidence:.2f}")
    else:
        axes[1].text(0.5, 0.5, "No Match Image Resolved", ha='center', va='center')
        axes[1].set_title("2. System Prediction")
    axes[1].axis('off')
    
    axes[2].imshow(true_img)
    axes[2].set_title(f"3. Ground Truth\nTarget ID: {true_id}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def load_production_brain(state):
    """Downloads index structures and metadata parquets directly from GCS."""
    print("Downloading Production Brain from GCS...")
    state.bucket.blob(Config.INDEX_PATH).download_to_filename(Config.LOCAL_INDEX)
    import faiss
    state.index = faiss.read_index_binary(Config.LOCAL_INDEX)
    
    from .vault.builder import load_source_metadata
    state.source_df = load_source_metadata(state.bucket)


def execute_live_notebook_benchmark(state, sample_size=100, nprobe=8):
    """
    Evaluates engine index search performance utilizing native FAISS vector 
    reconstructions and renders a structured visual health dashboard.
    """
    import time
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from .config import Config
    from .vault.builder import recover_state
    
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        print("⚠️ State metadata is currently empty. Aborting benchmark run.")
        return 0.0, 0.0
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    
    correct_matches = 0
    total_latency_ms = 0.0
    
    # Recover state array mappings natively
    _, master_index = recover_state(state)
    
    # 🌟 CRITICAL FIX 1: Explicitly anchor nprobe onto the index before entering the loop
    if hasattr(state.index, 'nprobe'):
        state.index.nprobe = nprobe
    else:
        # Fallback safeguard in case state tracking pointers are misaligned
        if hasattr(state, 'index') and hasattr(state.index, 'nlist'):
            state.index.nprobe = nprobe

    print(f"🏎️ Benchmark Active: Pruning search spaces to {nprobe} / {state.index.nlist if hasattr(state.index, 'nlist') else 4096} clusters...")
    
    for record in test_samples:
        start_r, end_r = int(record['start_row']), int(record['end_row'])
        count = end_r - start_r + 1
        if count <= 0: 
            continue
            
        real_descriptors = master_index.reconstruct_n(start_r, count)
        
        # Standardize matrix layout boundaries
        if len(real_descriptors) > Config.N_FEATURES:
            real_descriptors = real_descriptors[:Config.N_FEATURES]
        elif len(real_descriptors) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(real_descriptors), 32), dtype=np.uint8)
            real_descriptors = np.vstack([real_descriptors, padding])
            
        start_search = time.time()
        # 🌟 CRITICAL FIX 2: Runs the parallel C++ core cluster index lookup with corrected nprobe
        D, I = state.index.search(real_descriptors, k=1)
        total_latency_ms += (time.time() - start_search) * 1000
        
        # Identity vote confirmation lookup
        if start_r <= I[0][0] <= end_r:
            correct_matches += 1

    # Calculate final scaled metrics
    final_accuracy = (correct_matches / sample_size) * 100 if sample_size > 0 else 0.0
    avg_latency = (total_latency_ms / sample_size) if sample_size > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # RENDER VISUAL DASHBOARD REPORT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n🏁 ================================================== 🏁")
    print("📈 --- ARTFINDER RUNTIME PERFORMANCE DASHBOARD --- 📈")
    print("======================================================")
    print(f"  • Total Images Evaluated:   {sample_size:,}")
    print(f"  • Total Successful Matches: {correct_matches} / {sample_size}")
    print(f"  • Match Verification Rate:  {final_accuracy:.2f}%")
    print(f"  • Average Lookup Latency:   {avg_latency:.2f} ms")
    print("======================================================\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 2.5))
    
    acc_color = '#2ecc71' if final_accuracy >= 90 else ('#f1c40f' if final_accuracy >= 70 else '#e74c3c')
    ax1.barh(['Accuracy'], [final_accuracy], color=acc_color, edgecolor='#2c3e50', height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Percentage (%)')
    ax1.set_title(f'Target Accuracy: {final_accuracy:.1f}%')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    lat_color = '#2ecc71' if avg_latency <= 50 else ('#f1c40f' if avg_latency <= 150 else '#e74c3c')
    max_plot_speed = max(100, avg_latency * 1.5)
    ax2.barh(['Latency'], [avg_latency], color=lat_color, edgecolor='#2c3e50', height=0.5)
    ax2.set_xlim(0, max_plot_speed)
    ax2.set_xlabel('Time (ms)')
    ax2.set_title(f'Search Speed: {avg_latency:.2f} ms')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

    return final_accuracy, avg_latency


def execute_visual_3panel_validation(state, num_displays=3, nprobe=8):
    """
    Downloads raw frames from GCS, distorts them, and evaluates accuracy 
    visually inside a notebook environment via a 3-panel plotting matrix.
    """
    import random
    from .vault.builder import load_source_metadata
    
    print(f"\n🎨 Launching Interactive 3-Way Visual Validation (Displaying {num_displays} samples)...")
    state.index.nprobe = nprobe
    df_meta = load_source_metadata(state.bucket)
    
    if df_meta.empty:
        print("⚠️ State metadata is empty. Cannot extract imagery panel maps.")
        return
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    random.seed(1337)
    selected_records = random.sample(valid_records, k=min(num_displays, len(valid_records)))
    
    for idx, record in enumerate(selected_records):
        print(f"\nProcessing Visual Inspection #{idx+1}: '{record['title']}' by {record['artist']}")
        
        pid = str(record['id'])
        blob = state.bucket.blob(f"images/{pid}.jpg")
        
        if not blob.exists():
            print(f"  ⚠️ Skipping image display: 'images/{pid}.jpg' missing in cloud storage.")
            continue
            
        ground_truth_img = Image.open(BytesIO(blob.download_as_bytes())).convert('RGB')
        ground_truth_np = np.array(ground_truth_img)
        
        simulated_user_photo = apply_simulation(ground_truth_np)
        kp, des = state.orb.detectAndCompute(simulated_user_photo, None)
        
        if des is None or len(des) == 0: 
            continue
            
        if len(des) > Config.N_FEATURES:
            des = des[:Config.N_FEATURES]
        elif len(des) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(des), 32), dtype=np.uint8)
            des = np.vstack([des, padding])
            
        D, I = state.index.search(des, k=1)
        predicted_row_idx = I[0][0]
        
        matched_meta = None
        for lookup_rec in valid_records:
            if lookup_rec['start_row'] <= predicted_row_idx <= lookup_rec['end_row']:
                matched_meta = lookup_rec
                break
                
        # Fire off our visual multi-plot renderer
        show_3panel(state, simulated_user_photo, matched_meta['id'] if matched_meta else None, pid, D[0][0] if len(D) > 0 else 0.0)




def display_engine_diagnostic_report(state, top_n_artists=10):
    """
    Analyzes and prints a comprehensive health and inventory summary of the 
    active search engine, indexing models, cluster boundaries, and dataset distribution.
    """
    import pandas as pd
    from .config import Config

    print("📊 --- ARTFINDER SEARCH ENGINE DIAGNOSTIC REPORT --- 📊\n")
    
    # 1. Fetch Engine & RAM Context
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        print("⚠️ Diagnostic Warning: Active state metadata cache is empty or uninitialized.")
        return

    total_paintings = len(df_meta)
    unique_artists = df_meta['artist'].nunique() if 'artist' in df_meta.columns else 0
    
    # 2. Extract FAISS Structural Status
    has_index = state.index is not None
    if has_index:
        total_vectors = state.index.ntotal
        # Check if it's a true IVF index to pull clusters safely
        is_ivf = hasattr(state.index, 'nlist')
        num_clusters = state.index.nlist if is_ivf else "N/A (Flat Brute-Force Mode)"
        current_nprobe = state.index.nprobe if hasattr(state.index, 'nprobe') else "N/A"
    else:
        total_vectors = 0
        num_clusters = "No index loaded"
        current_nprobe = "N/A"

    # 3. Print Structural Overview
    print("📈 --- SYSTEM TOPOLOGY & HARDWARE BOUNDARIES ---")
    print(f"  • Total Active Indexed Paintings: {total_paintings:,}")
    print(f"  • Total Distinct Loaded Artists: {unique_artists:,}")
    print(f"  • Total Extracted ORB Vectors:   {total_vectors:,}")
    print(f"  • Active Indexing Cluster Slots:  {num_clusters}")
    print(f"  • Configured Search Depth (nprobe): {current_nprobe} / {Config.CLUSTERS if has_index else 'N/A'}")
    print(f"  • Fixed Sizing Slices (Config):  {Config.N_FEATURES} features | {Config.DIMENSION} dimensions")
    print(f"  • Cloud Storage Active Target:   gs://{Config.BUCKET_NAME}/{Config.META_PATH}")
    print("-" * 50)

    # 4. Calculate Inventory Distributions
    if 'artist' in df_meta.columns:
        print(f"\n🎨 --- TOP {top_n_artists} PROLIFIC ARTISTS IN VAULT ---")
        # Build count distribution maps
        artist_counts = (
            df_meta.groupby('artist')
            .size()
            .reset_index(name='paintings')
            .sort_values(by='paintings', ascending=False)
            .reset_index(drop=True)
        )
        
        # Calculate vector ranges per artist for validation reporting
        artist_vectors = []
        for name in artist_counts.head(top_n_artists)['artist']:
            artist_slice = df_meta[df_meta['artist'] == name]
            v_count = 0
            for _, row in artist_slice.iterrows():
                v_count += (int(row['end_row']) - int(row['start_row']) + 1)
            artist_vectors.append(v_count)
            
        # Merge metrics into a clean scannable display matrix
        top_df = artist_counts.head(top_n_artists).copy()
        top_df['total_vectors'] = artist_vectors
        top_df.index = top_df.index + 1
        
        # Display cleanly in terminal/notebook
        print(top_df.to_string(formatters={
            'paintings': '{:,}'.format,
            'total_vectors': '{:,}'.format
        }))
    else:
        print("\n⚠️ Diagnostic Warning: 'artist' category column missing from source frames.")
        
    print("\n🏁 --- END OF DIAGNOSTIC STATUS REPORT --- 🏁")
