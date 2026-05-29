# artfinder/evaluator.py
import time
import random
import cv2
import faiss
import numpy as np
import matplotlib.pyplot as plt
from .config import Config
from .searcher import ArtSearchEngine

# ──────────────────────────────────────────────────────────────────────────────
# 1. CORE ENGINE LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_production_brain(state):
    """Loads and mounts the remote production IVF models from GCS storage."""
    print("🧠 Downloading Production Brain from GCS...")
    
    # 1. Load the metadata safely
    from .vault.builder import load_source_metadata
    state.source_df = load_source_metadata(state.bucket)
    
    # 2. Download ONLY the fast IVF cluster index to memory
    blob = state.bucket.blob(Config.INDEX_PATH)
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_INDEX)
        state.index = faiss.read_index_binary(Config.LOCAL_INDEX)
        print(f"  ✅ Brain loaded. Active Metadata Records: {len(state.source_df):,}")
    else:
        print("  ⚠️ IVF Index not found in GCS! Cannot load brain.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. PURE MATHEMATICAL BENCHMARKS (Speed & Accuracy)
# ──────────────────────────────────────────────────────────────────────────────

def execute_live_notebook_benchmark(state, sample_size=100, nprobe=8, verbose=True):
    """Evaluates system search accuracy and latency using true dynamic C++ matrix batching."""
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        if verbose: print("⚠️ State metadata is empty. Aborting benchmark.")
        return 0.0, 0.0
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    search_engine = ArtSearchEngine(state)
    
    if verbose: print("📦 Downloading isolated raw vault for test queries...")
    state.bucket.blob(Config.VAULT_PATH).download_to_filename(Config.LOCAL_VAULT)
    flat_vault_index = faiss.read_index_binary(Config.LOCAL_VAULT)
    
    # Ensure nprobe depth is applied securely through the IDMap wrapper
    if hasattr(search_engine.state.index, 'index'):
        search_engine.state.index.index.nprobe = nprobe
    elif hasattr(search_engine.state.index, 'nprobe'):
        search_engine.state.index.nprobe = nprobe
        
    if verbose: print(f"🏎️ Benchmark Active: Compiling Dynamic Matrix for {sample_size} samples...")
    
    query_blocks = []
    offsets = []
    current_offset = 0
    
    # 1. Compile the master query matrix (NO ZERO PADDING)
    for record in test_samples:
        start_r, end_r = int(record['start_row']), int(record['end_row'])
        count = end_r - start_r + 1
        if count <= 0: continue
            
        real_descriptors = flat_vault_index.reconstruct_n(start_r, count)
        
        # Cap at max features, but DO NOT pad if under
        if len(real_descriptors) > Config.N_FEATURES:
            real_descriptors = real_descriptors[:Config.N_FEATURES]
            
        query_blocks.append(real_descriptors)
        
        # Track the exact start and stop indices for this specific image
        block_length = len(real_descriptors)
        offsets.append((current_offset, current_offset + block_length, record))
        current_offset += block_length
        
    if not query_blocks:
        return 0.0, 0.0
        
    # Stack the unpadded blocks into one lean master matrix
    master_query_matrix = np.vstack(query_blocks)
    
    # 2. Fire the single parallel batch execution
    start_search = time.time()
    D, I = search_engine.state.index.search(master_query_matrix, k=1)
    total_latency_ms = (time.time() - start_search) * 1000
    
    avg_latency = total_latency_ms / len(offsets)
    
    # 3. Tally matches using the precise offsets
    correct_matches = 0
    for start_idx, end_idx, record in offsets:
        # Extract only the exact results belonging to this image
        block_I = I[start_idx:end_idx]
        
        identity_tally = {}
        for row_idx in block_I.flatten():
            if row_idx in search_engine.row_to_metadata_map:
                item = search_engine.row_to_metadata_map[row_idx]
                identity_tally[item['id']] = identity_tally.get(item['id'], 0) + 1
                
        if identity_tally:
            predicted_id = max(identity_tally, key=identity_tally.get)
            if predicted_id == record['id']:
                correct_matches += 1

    final_accuracy = (correct_matches / len(offsets)) * 100 if offsets else 0.0

    if verbose:
        print("\n🏁 ================================================== 🏁")
        print("📈 --- ARTFINDER RUNTIME PERFORMANCE DASHBOARD --- 📈")
        print("======================================================")
        print(f"  • Total Images Evaluated:   {len(offsets):,}")
        print(f"  • Total Successful Matches: {correct_matches} / {len(offsets)}")
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


# ──────────────────────────────────────────────────────────────────────────────
# 3. ENVIRONMENTAL STRESS TESTS & VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_wall_photo(img_np):
    """Shrinks the artwork and places it on a randomized, noisy background."""
    h, w = img_np.shape[:2]
    
    scale = 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    painting = cv2.resize(img_np, (new_w, new_h))
    
    bg_color = [random.randint(40, 220) for _ in range(3)]
    wall = np.full((h, w, 3), bg_color, dtype=np.uint8)
    
    noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    wall = np.clip(wall + noise, 0, 255).astype(np.uint8)
    
    y_offset = (h - new_h) // 2
    x_offset = (w - new_w) // 2
    wall[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = painting
    
    return wall


def _simulate_book_page(img_np):
    """Warps the 2D image over a 3D cylindrical curve with spine shading."""
    h, w = img_np.shape[:2]

    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map = x_map.astype(np.float32)
    y_map = y_map.astype(np.float32)

    amplitude = h * 0.05 
    norm_x = x_map / w
    y_map = y_map - (amplitude * np.sin(norm_x * np.pi))
    y_map = y_map + amplitude

    paper_color = (240, 245, 245)
    warped_page = cv2.remap(
        img_np, x_map, y_map, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=paper_color
    )

    shadow_gradient = 0.4 + 0.6 * np.power(norm_x, 0.6) 
    shadow_gradient = shadow_gradient[:, :, np.newaxis] 
    warped_page = np.clip(warped_page * shadow_gradient, 0, 255).astype(np.uint8)

    bg_color = [random.randint(120, 160) for _ in range(3)]
    desk = np.full((h + int(amplitude*2), w + 40, 3), bg_color, dtype=np.uint8)   
 
    y_offset = int(amplitude)
    x_offset = 20
    desk[y_offset:y_offset+h, x_offset:x_offset+w] = warped_page

    return desk


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


def run_environmental_stress_test(state, sample_size=10, visualize_top_n=3, nprobe=8):
    """Tests the engine's resilience against non-linear geometric and environmental noise."""
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        print("⚠️ State metadata is empty. Aborting benchmark.")
        return
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    random.seed(int(time.time()))
    test_samples = random.sample(valid_records, k=sample_size)
    
    search_engine = ArtSearchEngine(state)
    correct_matches = 0
    latencies = []
    
    print(f"🌪️ Initiating Environmental Stress Test ({sample_size} samples)...")
    
    for idx, record in enumerate(test_samples):
        artwork_id = record['id']
        
        # 1. Fetch the raw original image from GCS
        try:
            blob = state.bucket.blob(f"images/{artwork_id}.jpg")
            img_bytes = blob.download_as_bytes()
            nparr = np.frombuffer(img_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"⚠️ Failed to fetch {artwork_id}: {e}")
            continue
            
        # 2. Randomly select an environmental hazard
        scenario = random.choice(["Wall", "Book"])
        if scenario == "Wall":
            mutated_img = _simulate_wall_photo(original_img)
        else:
            mutated_img = _simulate_book_page(original_img)
            
        # 3. Run the live search engine on the mutated image
        start_time = time.time()
        result = search_engine.find_match(mutated_img, nprobe=nprobe)
        latencies.append((time.time() - start_time) * 1000)
        
        # 4. Check accuracy
        is_correct = (result.artwork_id == artwork_id)
        if is_correct:
            correct_matches += 1
            
        # 5. Render the 3-panel visualizer
        if idx < visualize_top_n:
            status = "✅ SUCCESS" if is_correct else f"❌ FAILED (Matched: {result.artwork_id})"
            print(f"\n--- Test {idx+1}: {status} [{scenario} Scenario] ---")
            
            if result.artwork_id != "unknown":
                visualize_orb_matches(mutated_img, result, state)
            else:
                print("Engine returned 'Unknown' - No visual to render.")

    final_accuracy = (correct_matches / len(test_samples)) * 100 if test_samples else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0
    
    print("\n🏁 ================================================== 🏁")
    print("🌪️ --- ENVIRONMENTAL STRESS TEST RESULTS --- 🌪️")
    print("======================================================")
    print(f"  • Images Mutated & Tested:  {len(test_samples)}")
    print(f"  • Successful Matches:       {correct_matches} / {len(test_samples)}")
    print(f"  • Noise Survival Rate:      {final_accuracy:.2f}%")
    print(f"  • Average Lookup Latency:   {avg_latency:.2f} ms")
    print("======================================================\n")
