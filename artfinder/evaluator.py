import time
import random
import numpy as np
import matplotlib.pyplot as plt
from .config import Config
from .vault.builder import recover_state
from .searcher import ArtSearchEngine

def load_production_brain(state):
    """Loads and mounts the remote production vault models from GCS storage."""
    print("🧠 Downloading Production Brain from GCS...")
    recover_state(state)
    print(f"  ✅ Brain loaded. Active Metadata Records: {len(state.source_df):,}")


def execute_live_notebook_benchmark(state, sample_size=100, nprobe=8):
    """
    Evaluates system search accuracy and latency profiles using the standalone
    ArtSearchEngine module, rendering a clean performance dashboard.
    """
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        print("⚠️ State metadata is empty. Aborting benchmark.")
        return 0.0, 0.0
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    
    # 🌟 INITIALIZE THE INDEPENDENT ENGINE
    # This engine automatically picks up the fast clustered state.index
    search_engine = ArtSearchEngine(state)
    
    # We still need the flat vault locally strictly to extract authentic 
    # vector blocks to simulate an image query input.
    _, flat_vault_index = recover_state(state)
    
    correct_matches = 0
    latencies = []
    
    print(f"🏎️ Benchmark Active: Running performance passes over {sample_size} samples...")
    
    for record in test_samples:
        start_r, end_r = int(record['start_row']), int(record['end_row'])
        count = end_r - start_r + 1
        if count <= 0: continue
            
        # Extract authentic vector blocks to simulate an image query input
        real_descriptors = flat_vault_index.reconstruct_n(start_r, count)
        if len(real_descriptors) > Config.N_FEATURES:
            real_descriptors = real_descriptors[:Config.N_FEATURES]
        elif len(real_descriptors) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(real_descriptors), 32), dtype=np.uint8)
            real_descriptors = np.vstack([real_descriptors, padding])
            
        # 🚀 EXECUTE INDEPENDENT PRODUCTION LOOKUP
        # The engine natively uses its fast clustered index, bringing latency back to ms
        start_search = time.time()
        D, I = search_engine.state.index.search(real_descriptors, k=1)
        latency_ms = (time.time() - start_search) * 1000
        latencies.append(latency_ms)
        
        # Tally matches via the search engine's internal map
        identity_tally = {}
        for row_idx in I.flatten():
            if row_idx in search_engine.row_to_metadata_map:
                item = search_engine.row_to_metadata_map[row_idx]
                identity_tally[item['id']] = identity_tally.get(item['id'], 0) + 1
                
        if identity_tally:
            predicted_id = max(identity_tally, key=identity_tally.get)
            if predicted_id == record['id']:
                correct_matches += 1

    # Calculate metrics
    final_accuracy = (correct_matches / sample_size) * 100 if sample_size > 0 else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0

    # Render dashboard report charts
    print("\n🏁 ================================================== 🏁")
    print("📈 --- ARTFINDER RUNTIME PERFORMANCE DASHBOARD --- 📈")
    print("======================================================")
    print(f"  • Total Images Evaluated:   {sample_size:,}")
    print(f"  • Total Successful Matches: {correct_matches} / {sample_size}")
    print(f"  • Match Verification Rate:  {final_accuracy:.2f}%")
    print(f"  • Average Lookup Latency:   {avg_latency:.2f} ms")
    print("======================================================\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 2.5))
    acc_color = '#2ecc71' if final_accuracy >= 90 else '#e74c3c'
    ax1.barh(['Accuracy'], [final_accuracy], color=acc_color, edgecolor='#2c3e50', height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_title(f'True Accuracy: {final_accuracy:.1f}%')
    ax1.grid(axis='x', linestyle='--', alpha=0.5)

    lat_color = '#2ecc71' if avg_latency <= 50 else '#e74c3c'
    ax2.barh(['Latency'], [avg_latency], color=lat_color, edgecolor='#2c3e50', height=0.5)
    ax2.set_xlim(0, max(50, avg_latency * 1.5))
    ax2.set_title(f'Search Speed: {avg_latency:.2f} ms')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

    return final_accuracy, avg_latency
