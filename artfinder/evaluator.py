# artfinder/evaluator.py
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import faiss
from .config import Config
from .searcher import ArtSearchEngine

def load_production_brain(state):
    """Loads and mounts the remote production IVF models from GCS storage."""
    print("🧠 Downloading Production Brain from GCS...")
    
    from .vault.builder import load_source_metadata
    state.source_df = load_source_metadata(state.bucket)
    
    blob = state.bucket.blob(Config.INDEX_PATH)
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_INDEX)
        state.index = faiss.read_index_binary(Config.LOCAL_INDEX)
        print(f"  ✅ Brain loaded. Active Metadata Records: {len(state.source_df):,}")
    else:
        print("  ⚠️ IVF Index not found in GCS! Cannot load brain.")



def execute_live_notebook_benchmark(state, sample_size=100, nprobe=8, verbose=True):
    """Evaluates system search accuracy and latency using C++ matrix batching."""
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        if verbose: print("⚠️ State metadata is empty. Aborting benchmark.")
        return 0.0, 0.0
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    import random
    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    search_engine = ArtSearchEngine(state)
    
    if verbose: print("📦 Downloading isolated raw vault for test queries...")
    state.bucket.blob(Config.VAULT_PATH).download_to_filename(Config.LOCAL_VAULT)
    import faiss
    flat_vault_index = faiss.read_index_binary(Config.LOCAL_VAULT)
    
    # Ensure nprobe depth is applied securely through the IDMap wrapper
    if hasattr(search_engine.state.index, 'index'):
        search_engine.state.index.index.nprobe = nprobe
    elif hasattr(search_engine.state.index, 'nprobe'):
        search_engine.state.index.nprobe = nprobe
        
    if verbose: print(f"🏎️ Benchmark Active: Compiling C++ Matrix for {sample_size} samples...")
    
    query_blocks = []
    valid_test_samples = []
    
    # 1. Compile the master query matrix (Eliminate Python Loop Overhead)
    for record in test_samples:
        start_r, end_r = int(record['start_row']), int(record['end_row'])
        count = end_r - start_r + 1
        if count <= 0: continue
            
        real_descriptors = flat_vault_index.reconstruct_n(start_r, count)
        if len(real_descriptors) > Config.N_FEATURES:
            real_descriptors = real_descriptors[:Config.N_FEATURES]
        elif len(real_descriptors) < Config.N_FEATURES:
            import numpy as np
            padding = np.zeros((Config.N_FEATURES - len(real_descriptors), 32), dtype=np.uint8)
            real_descriptors = np.vstack([real_descriptors, padding])
            
        query_blocks.append(real_descriptors)
        valid_test_samples.append(record)
        
    if not query_blocks:
        return 0.0, 0.0
        
    import numpy as np
    import time
    master_query_matrix = np.vstack(query_blocks)
    
    # 2. Fire the single parallel batch execution
    start_search = time.time()
    D, I = search_engine.state.index.search(master_query_matrix, k=1)
    total_latency_ms = (time.time() - start_search) * 1000
    
    # 3. Calculate true amortized average
    avg_latency = total_latency_ms / len(valid_test_samples)
    
    # 4. Tally matches block by block
    correct_matches = 0
    for i, record in enumerate(valid_test_samples):
        # Extract the 500 specific results for this single image
        block_I = I[i * Config.N_FEATURES : (i + 1) * Config.N_FEATURES]
        
        identity_tally = {}
        for row_idx in block_I.flatten():
            if row_idx in search_engine.row_to_metadata_map:
                item = search_engine.row_to_metadata_map[row_idx]
                identity_tally[item['id']] = identity_tally.get(item['id'], 0) + 1
                
        if identity_tally:
            predicted_id = max(identity_tally, key=identity_tally.get)
            if predicted_id == record['id']:
                correct_matches += 1

    final_accuracy = (correct_matches / len(valid_test_samples)) * 100 if valid_test_samples else 0.0

    if verbose:
        print("\n🏁 ================================================== 🏁")
        print("📈 --- ARTFINDER RUNTIME PERFORMANCE DASHBOARD --- 📈")
        print("======================================================")
        print(f"  • Total Images Evaluated:   {len(valid_test_samples):,}")
        print(f"  • Total Successful Matches: {correct_matches} / {len(valid_test_samples)}")
        print(f"  • Match Verification Rate:  {final_accuracy:.2f}%")
        print(f"  • Average Lookup Latency:   {avg_latency:.2f} ms")
        print("======================================================\n")

    return final_accuracy, avg_latency



def run_scaling_stress_test(state, n_sizes=[10, 50, 100, 250, 500]):
    """Runs the benchmark across scaling input sizes to verify cluster O(1) latency."""
    print("🚀 Initiating N-Size Scaling Test...")
    accuracies = []
    latencies = []

    for size in n_sizes:
        print(f"🧪 Testing Sample Size: {size}...")
        # Run silently to avoid spamming the console
        acc, lat = execute_live_notebook_benchmark(state, sample_size=size, verbose=False)
        accuracies.append(acc)
        latencies.append(lat)

    # Plot the scaling results
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Sample Size (N)')
    ax1.set_ylabel('Average Latency (ms)', color=color)
    ax1.plot(n_sizes, latencies, marker='o', color=color, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(latencies) * 1.5)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(n_sizes, accuracies, marker='s', color=color, linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)

    plt.title("Engine Scaling Performance (IVF Cluster Validation)")
    fig.tight_layout()
    plt.show()


def visualize_orb_matches(query_img, match_result, state):
    """Downloads the matched image from GCS and draws the visual point connections."""
    print(f"\n🖼️ Fetching matched asset 'gs://{state.bucket.name}/images/{match_result.artwork_id}.jpg'...")
    
    try:
        blob = state.bucket.blob(f"images/{match_result.artwork_id}.jpg")
        img_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(img_bytes, np.uint8)
        matched_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        query_resized = cv2.resize(query_img, Config.RESIZE_DIM)
        match_resized = cv2.resize(matched_img, Config.RESIZE_DIM)
        
        kp1, des1 = state.orb.detectAndCompute(query_resized, None)
        kp2, des2 = state.orb.detectAndCompute(match_resized, None)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        img_matches = cv2.drawMatches(
            query_resized, kp1, 
            match_resized, kp2, 
            matches[:50], None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(18, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Match: {match_result.title} by {match_result.artist} | Confidence: {match_result.confidence:.2%}")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not render visual match: {e}")
