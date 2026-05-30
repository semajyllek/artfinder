import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from .config import Config
from .evaluator import load_production_brain
from .vault.builder import save_source_metadata, process_image

# ──────────────────────────────────────────────────────────────────────────────
# 1. CORE PIPELINE FUNCTIONS (The Atomic Steps)
# ──────────────────────────────────────────────────────────────────────────────

def _get_starting_state(state, is_rebuild=False):
    """Returns the starting Flat Vault, Metadata, and known IDs."""
    if is_rebuild:
        print("🧹 REBUILD MODE: Initializing blank slate...")
        # 1. Purge GCS (Keep it clean!)
        blobs = state.bucket.list_blobs(prefix="images/")
        for blob in blobs: blob.delete()
        
        # 2. Create Empty States
        flat_vault = faiss.IndexBinaryFlat(256)
        df_meta = pd.DataFrame(columns=['id', 'artist', 'title', 'start_row', 'end_row'])
        return flat_vault, df_meta, set()
    else:
        print("📦 APPEND MODE: Downloading existing Flat Vault & Metadata...")
        load_production_brain(state)
        state.bucket.blob(Config.VAULT_PATH).download_to_filename(Config.LOCAL_VAULT)
        
        flat_vault = faiss.read_index_binary(Config.LOCAL_VAULT)
        df_meta = state.source_df if state.source_df is not None else pd.DataFrame()
        existing_ids = set(df_meta['id'].tolist()) if not df_meta.empty else set()
        
        print(f"   • Current Vault Vectors: {flat_vault.ntotal:,}")
        return flat_vault, df_meta, existing_ids


def _extract_source_features(state, limit, existing_ids, current_total_rows):
    """Streams the dataset, extracts ORB features, and tracks matrix offsets."""
    print(f"📥 Extracting features (Target: {limit} new unique records)...")
    dataset = load_dataset("huggan/wikiart", split="train", streaming=True)
    
    new_vectors = []
    new_metadata_records = []
    processed_count = 0
    
    for item in tqdm(dataset, total=limit, desc="Extracting"):
        if processed_count >= limit:
            break
            
        artwork_id = item.get('id', str(hash(item['image'].tobytes())))
        
        if artwork_id in existing_ids:
            continue
            
        image = item['image'].convert('RGB')
        vectors = process_image(image, state.orb)
        
        if vectors is not None and len(vectors) > 0:
            start_row = current_total_rows + len(new_vectors)
            end_row = start_row + len(vectors) - 1
            
            new_vectors.extend(vectors)
            new_metadata_records.append({
                'id': artwork_id,
                'artist': item.get('artist', 'Unknown'),
                'title': item.get('title', 'Unknown'),
                'start_row': start_row,
                'end_row': end_row
            })
            
            # Optional: Save original image to GCS for the 3-Panel Visualizer
            img_blob = state.bucket.blob(f"images/{artwork_id}.jpg")
            if not img_blob.exists():
                image.save("temp.jpg", format="JPEG", quality=85)
                img_blob.upload_from_filename("temp.jpg")
                
            processed_count += 1
            
    return new_vectors, new_metadata_records


def _commit_raw_storage(state, flat_vault, df_meta, new_vectors, new_records):
    """Appends data to the raw Vault and Parquet, then uploads to GCS."""
    print("\n💾 Committing new data to Raw Storage...")
    
    # Update Flat Vault
    matrix_to_add = np.array(new_vectors, dtype=np.uint8)
    flat_vault.add(matrix_to_add)
    faiss.write_index_binary(flat_vault, Config.LOCAL_VAULT)
    state.bucket.blob(Config.VAULT_PATH).upload_from_filename(Config.LOCAL_VAULT)
    print(f"   • Updated Vault Vectors: {flat_vault.ntotal:,}")
    
    # Update Metadata
    new_df = pd.DataFrame(new_records)
    updated_df = pd.concat([df_meta, new_df], ignore_index=True) if not df_meta.empty else new_df
    save_source_metadata(updated_df, state.bucket)
    
    return flat_vault


def _train_production_brain(state, flat_vault):
    """Rebuilds the fast IVF search clusters from the master Flat Vault."""
    print("\n🧠 Retraining the IVF Cluster Brain...")
    
    total_vectors = flat_vault.ntotal
    master_matrix = flat_vault.reconstruct_n(0, total_vectors)
    
    # Dynamic Voronoi Math (Scales clusters based on dataset size)
    nlist = max(100, int(4 * np.sqrt(total_vectors))) 
    
    quantizer = faiss.IndexBinaryFlat(256)
    new_ivf_index = faiss.IndexBinaryIVF(quantizer, 256, nlist)
    
    print(f"   • Training {nlist} Voronoi centroids...")
    new_ivf_index.train(master_matrix)
    
    print("   • Injecting explicit row coordinates...")
    row_ids = np.arange(total_vectors)
    new_ivf_index.add_with_ids(master_matrix, row_ids)
    
    print("📤 Uploading new Production Brain to GCS...")
    faiss.write_index_binary(new_ivf_index, Config.LOCAL_INDEX)
    state.bucket.blob(Config.INDEX_PATH).upload_from_filename(Config.LOCAL_INDEX)
    
    state.index = new_ivf_index
    print("✅ Brain successfully trained and deployed!")


# ──────────────────────────────────────────────────────────────────────────────
# 2. THE IDEMPOTENT ORCHESTRATORS (Your 1-Line Commands)
# ──────────────────────────────────────────────────────────────────────────────

def run_incremental_update(state, limit=1000):
    """Appends new unique images to the existing engine and retrains the clusters."""
    print(f"🚀 --- STARTING INCREMENTAL UPDATE (Limit: {limit}) --- 🚀\n")
    
    flat_vault, df_meta, existing_ids = _get_starting_state(state, is_rebuild=False)
    
    new_vecs, new_recs = _extract_source_features(state, limit, existing_ids, current_total_rows=flat_vault.ntotal)
    
    if not new_vecs:
        print("\n⚠️ No new unique records found. Engine is already up to date.")
        return
        
    updated_vault = _commit_raw_storage(state, flat_vault, df_meta, new_vecs, new_recs)
    _train_production_brain(state, updated_vault)
    print("\n🏆 --- INCREMENTAL UPDATE COMPLETE --- 🏆")


def run_complete_rebuild(state, limit=1000):
    """Wipes the database entirely and builds a fresh engine from scratch."""
    print(f"⚠️ --- STARTING COMPLETE REBUILD (Limit: {limit}) --- ⚠️\n")
    
    flat_vault, df_meta, existing_ids = _get_starting_state(state, is_rebuild=True)
    
    new_vecs, new_recs = _extract_source_features(state, limit, existing_ids, current_total_rows=0)
    
    if not new_vecs:
        print("\n⚠️ Failed to extract any features. Rebuild aborted.")
        return
        
    updated_vault = _commit_raw_storage(state, flat_vault, df_meta, new_vecs, new_recs)
    _train_production_brain(state, updated_vault)
    print("\n🏆 --- COMPLETE REBUILD SUCCESSFUL --- 🏆")
