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
        from .ingestor import load_source_metadata
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

    def _run_inference(self, img_np, nprobe=8):
        start = time.time()
        self.state.index.nprobe = nprobe
        resized = cv2.resize(img_np, Config.RESIZE_DIM)
        kp, des = self.state.orb.detectAndCompute(resized, None)
        
        if des is None or len(des) == 0:
            return None, 0.0, (time.time() - start)*1000
            
        if len(des) > Config.N_FEATURES:
            des = des[:Config.N_FEATURES]
        elif len(des) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(des), 32), dtype=np.uint8)
            des = np.vstack([des, padding])

        D, I = self.state.index.search(des, k=1)
        latency = (time.time() - start) * 1000
        
        # Resolve identity frequency matches
        counts = {}
        for row_idx in I.flatten():
            if row_idx < 0: continue
            for _, r in self.state.source_df.iterrows():
                if r['start_row'] <= row_idx <= r['end_row']:
                    counts[r['id']] = counts.get(r['id'], 0) + 1
                    break
                    
        if not counts:
            return None, 0.0, latency
        best_id = max(counts, key=counts.get)
        confidence = counts[best_id] / Config.N_FEATURES
        return best_id, confidence, latency

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
    """Renders visual performance comparisons."""
    meta = None
    if pred_id:
        r = state.source_df[state.source_df['id'] == pred_id]
        if not r.empty: meta = r.iloc[0]
        
    try:
        blob = state.bucket.blob(f"images/{true_id}.jpg")
        true_img = Image.open(BytesIO(blob.download_as_bytes()))
    except: return

    pred_img = None
    if meta:
        try:
            p_blob = state.bucket.blob(f"images/{meta['id']}.jpg")
            pred_img = Image.open(BytesIO(p_blob.download_as_bytes()))
        except: pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(test_photo); axes[0].set_title("1. Simulated User Photo"); axes[0].axis('off')
    if pred_img:
        axes[1].imshow(pred_img)
        axes[1].set_title(f"2. Prediction: {meta['title']}\nConf: {confidence:.2f}")
    axes[1].axis('off')
    axes[2].imshow(true_img); axes[2].set_title(f"3. Ground Truth\n{true_id}"); axes[2].axis('off')
    plt.show()


def load_production_brain(state):
    """Downloads index structures and metadata parquets directly from GCS."""
    print("Downloading Production Brain from GCS...")
    state.bucket.blob(Config.INDEX_PATH).download_to_filename(Config.LOCAL_INDEX)
    import faiss
    state.index = faiss.read_index_binary(Config.LOCAL_INDEX)
    from .ingestor import load_source_metadata
    state.source_df = load_source_metadata(state.bucket)


def execute_live_notebook_benchmark(state, sample_size=100):
    """Evaluates indexes utilizing native vector reconstructions out of the FAISS arrays."""
    import time
    from .ingestor import recover_state
    df_meta = state.source_df
    if df_meta.empty:
        return 0.0, 0.0
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    
    correct_matches = 0
    total_latency_ms = 0.0
    
    _, master_index = recover_state(state)
    
    for record in test_samples:
        start_r, end_r = int(record['start_row']), int(record['end_row'])
        count = end_r - start_r + 1
        if count <= 0: continue
            
        real_descriptors = master_index.reconstruct_n(start_r, count)
        if len(real_descriptors) > Config.N_FEATURES:
            real_descriptors = real_descriptors[:Config.N_FEATURES]
        elif len(real_descriptors) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(real_descriptors), 32), dtype=np.uint8)
            real_descriptors = np.vstack([real_descriptors, padding])
            
        start_search = time.time()
        D, I = state.index.search(real_descriptors, k=1)
        total_latency_ms += (time.time() - start_search) * 1000
        
        if start_r <= I[0][0] <= end_r:
            correct_matches += 1

    return (correct_matches / sample_size) if sample_size > 0 else 0.0, (total_latency_ms / sample_size) if sample_size > 0 else 0.0
