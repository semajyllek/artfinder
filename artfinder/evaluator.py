import time
import random
import numpy as np
import matplotlib.pyplot as plt
import faiss
from .config import Config
from .searcher import ArtSearchEngine

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


def execute_live_notebook_benchmark(state, sample_size=100, nprobe=8):
    """Evaluates system search accuracy and latency profiles using the standalone module."""
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        print("⚠️ State metadata is empty. Aborting benchmark.")
        return 0.0, 0.0
        
    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))
    
    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    
    search_engine = ArtSearchEngine(state)
    
    # 🌟 THE FIX: Isolate the raw vectors without mutating the global state
    print("📦 Downloading isolated raw vault for test queries...")
    state.bucket.blob(Config.VAULT_PATH).download_to_filename(Config.LOCAL_VAULT)
    flat_vault_index = faiss.read_index_binary(Config.LOCAL_VAULT)
    
    correct_matches = 0
    latencies = []
    
    print(f"🏎️ Benchmark Active: Running performance passes over {sample_size} samples...")
    
    for record in test_samples:
        start_r, end_r = int(record['start_row']), int(record['end_row'])
        count = end_r - start_r + 1
        if count <= 0: continue
            
        real_descriptors = flat_vault_index.reconstruct_n(start_r, count)
        if len(real_descriptors) > Config.N_FEATURES:
            real_descriptors = real_descriptors[:Config.N_FEATURES]
        elif len(real_descriptors) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(real_descriptors), 32), dtype=np.uint8)
            real_descriptors = np.vstack([real_descriptors, padding])
            
        # Natively uses the fast clustered index. 
        # (Ensure you also applied the searcher.py nprobe fix from earlier!)
        start_search = time.time()
        D, I = search_engine.state.index.search(real_descriptors, k=1)
        latency_ms = (time.time() - start_search) * 1000
        latencies.append(latency_ms)
        
        identity_tally = {}
        for row_idx in I.flatten():
            if row_idx in search_engine.row_to_metadata_map:
                item = search_engine.row_to_metadata_map[row_idx]
                identity_tally[item['id']] = identity_tally.get(item['id'], 0) + 1
                
        if identity_tally:
            predicted_id = max(identity_tally, key=identity_tally.get)
            if predicted_id == record['id']:
                correct_matches += 1

    final_accuracy = (correct_matches / sample_size) * 100 if sample_size > 0 else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0

    print("\n🏁 ================================================== 🏁")
    print("📈 --- ARTFINDER RUNTIME PERFORMANCE DASHBOARD --- 📈")
    print("======================================================")
    print(f"  • Total Images Evaluated:   {sample_size:,}")
    print(f"  • Total Successful Matches: {correct_matches} / {sample_size}")
    print(f"  • Match Verification Rate:  {final_accuracy:.2f}%")
    print(f"  • Average Lookup Latency:   {avg_latency:.2f} ms")
    print("======================================================\n")

    return final_accuracy, avg_latency
