import os
import gc
import cv2
import faiss
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from ..config import Config

# ──────────────────────────────────────────────────────────────────────────────
# 1. CORE PARQUET & RECOVERY UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def load_source_metadata(bucket):
    """
    Downloads the active production metadata parquet index tracking sheet 
    directly from the mounted GCS bucket locations.
    """
    blob = bucket.blob(Config.META_PATH)
    if blob.exists():
        content = blob.download_as_bytes()
        return pd.read_parquet(BytesIO(content))
    return pd.DataFrame(columns=['id', 'title', 'artist', 'url', 'start_row', 'end_row'])


def recover_state(state):
    """
    Synchronizes local environment blocks by reading down historical parquets 
    and mounting unclustered flat binary vaults out of your GCS bucket.
    """
    source_df = load_source_metadata(state.bucket)
    blob = state.bucket.blob(Config.VAULT_PATH)
    
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_VAULT)
        master_index = faiss.read_index_binary(Config.LOCAL_VAULT)
    else:
        master_index = faiss.IndexBinaryFlat(Config.DIMENSION)
        
    state.source_df = source_df
    state.index = master_index
    return source_df, master_index


def vault_checkpoint(state, new_records, master_index):
    """
    Performs safe transactional serialization passes upstream to GCS to safeguard 
    extracted high-dimensional array states during heavy streaming routines.
    """
    if not new_records: 
        return
        
    current_source = load_source_metadata(state.bucket)
    updated_source = pd.concat([current_source, pd.DataFrame(new_records)], ignore_index=True)

    # Serialize metadata updates to GCS
    updated_source.to_parquet(Config.LOCAL_META, index=False)
    state.bucket.blob(Config.META_PATH).upload_from_filename(Config.LOCAL_META)

    # Serialize raw flat binary descriptors to GCS
    faiss.write_index_binary(master_index, Config.LOCAL_VAULT)
    state.bucket.blob(Config.VAULT_PATH).upload_from_filename(Config.LOCAL_VAULT)
    
    state.source_df = updated_source


# ──────────────────────────────────────────────────────────────────────────────
# 2. RUNTIME WORKSPACE CLEANERS
# ──────────────────────────────────────────────────────────────────────────────

def purge_local_cache_files():
    """Removes leftover configuration and cache artifacts from local storage."""
    print("🧹 Purging local system cache tracks...")
    local_targets = [Config.LOCAL_META, Config.LOCAL_VAULT, Config.LOCAL_INDEX]
    for filename in local_targets:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"  Deleted local track: {filename}")
            except OSError as e:
                print(f"  ⚠️ Could not clear local file {filename}: {e}")



def purge_gcs_production_vault(state):
    """Deletes existing binary vaults, parquets, and image assets inside the active cloud bucket."""
    print("🗑️ Erasing historical engine assets from Google Cloud Storage...")
    gcs_targets = [Config.META_PATH, Config.VAULT_PATH, Config.INDEX_PATH]
    for gcs_path in gcs_targets:
        blob = state.bucket.blob(gcs_path)
        if blob.exists():
            try:
                blob.delete()
                print(f"  Deleted from GCS Bucket: {gcs_path}")
            except Exception as e:
                print(f"  ⚠️ GCS deletion failure on path {gcs_path}: {e}")

    print("🗑️ Sweeping raw image assets...")
    try:
        blobs = state.bucket.list_blobs(prefix="images/")
        image_count = 0
        for blob in blobs:
            blob.delete()
            image_count += 1
        print(f"  ✅ Deleted {image_count:,} orphaned files from 'images/' prefix.")
    except Exception as e:
        print(f"  ⚠️ Failed to sweep images directory: {e}")



# ──────────────────────────────────────────────────────────────────────────────
# 3. CORE PROCESS WORKER FLIGHT CELL (Executed in isolated CPU cores)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features_worker(record_batch):
    """
    Isolated worker function that runs on an independent CPU core.
    Processes a localized chunk of records, completely bypassing the GIL.
    """
    # Re-instantiate local ORB descriptors per process workspace
    local_orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, WTA_K=2)
    processed_results = []
    
    for record in record_batch:
        try:
            img_np = record['image']
            
            # Defensive verification of data structure
            if not isinstance(img_np, np.ndarray):
                continue

            resized = cv2.resize(img_np, (300, 300))  # Matches Config.RESIZE_DIM
            kp, des = local_orb.detectAndCompute(resized, None)
            
            if des is not None and len(des) > 0:
                # Compress the image to JPEG bytes right inside the worker core
                # to offload overhead from the parent coordination process
                buffer = BytesIO()
                Image.fromarray(img_np).save(buffer, format="JPEG", quality=85)
                
                processed_results.append({
                    'visual_id': record['visual_id'],
                    'title': record['title'],
                    'artist': record['artist'],
                    'url': record.get('SourceURL', 'https://www.wikiart.org'),
                    'descriptors': des,
                    'image_bytes': buffer.getvalue()
                })
        except Exception:
            continue
            
    return processed_results


# ──────────────────────────────────────────────────────────────────────────────
# 4. PARALLEL BATCH INGESTION DRIVER
# ──────────────────────────────────────────────────────────────────────────────

class VaultBuilder:
    def __init__(self, state):
        self.state = state

    def ingest_stream(self, data_stream, batch_name, total_records=None):
        """
        Ingestion gateway optimized with true multi-processing.
        Chunks stream layers and maps feature computation across all available CPU cores.
        """
        _, master_index = recover_state(self.state)
        
        try:
            current_df = load_source_metadata(self.state.bucket)
            known_ids = set(current_df['id'].dropna().unique())
        except Exception:
            known_ids = set()

        cache = []
        total_scanned = 0
        total_matched = 0
        unique_artists = set()
        
        # Hardware topology configuration metrics
        num_cores = max(1, cpu_count() - 1)
        chunk_size = 64  # Size of task allocations pushed to individual workers
        accumulator = []

        print(f"🚀 Initializing parallel multi-core engine for batch: {batch_name}")
        print(f"💥 Spawning ProcessPool mapping layout across {num_cores} active CPU cores...")

        with Pool(processes=num_cores) as pool, tqdm(desc=f"Vaulting {batch_name}", total=total_records) as pbar:
            
            for record in data_stream:
                visual_id = record.get('visual_id')
                if visual_id in known_ids:
                    continue
                
                # Strip unpicklable objects out before crossing process boundary structures
                if isinstance(record['image'], Image.Image):
                    record['image'] = np.array(record['image'].convert('RGB'))
                
                accumulator.append(record)
                total_scanned += 1
                
                if 'artist' in record:
                    unique_artists.add(record['artist'])

                # When your accumulator allocation matches your core capacity, distribute workload
                if len(accumulator) >= (chunk_size * num_cores):
                    # Subdivide master batch into localized chunks
                    micro_chunks = [accumulator[i:i + chunk_size] for i in range(0, len(accumulator), chunk_size)]
                    
                    # Map task payloads to worker cores
                    parallel_outputs = pool.map(_extract_features_worker, micro_chunks)
                    
                    # Reduce Phase: Synchronized main thread collects outputs, updates FAISS index & GCS
                    for core_output in parallel_outputs:
                        for item in core_output:
                            total_matched += 1
                            start_row = master_index.ntotal
                            master_index.add(item['descriptors'])
                            
                            # Fire compressed payload to target cloud targets
                            filename = f"{item['visual_id']}.jpg"
                            blob = self.state.bucket.blob(f"images/{filename}")
                            blob.upload_from_string(item['image_bytes'], content_type='image/jpeg')
                            
                            cache.append({
                                'id': item['visual_id'],
                                'title': str(item['title']),
                                'artist': str(item['artist']),
                                'url': item['url'],
                                'start_row': start_row,
                                'end_row': master_index.ntotal - 1
                            })
                    
                    pbar.update(len(accumulator))
                    accumulator = []  # Clear memory allocations

                    # Evaluate checkpoint threshold intervals
                    if len(cache) >= Config.CHECKPOINT_SIZE:
                        vault_checkpoint(self.state, cache, master_index)
                        print(f"\n💾 Flushing safe checkpoint slice to GCS. Index length: {master_index.ntotal:,}")
                        cache = []
                        gc.collect()

                    # Render live terminal performance reports safely between lines
                    print(f"\n✨ --- PARALLEL DASHBOARD [Processed: {total_scanned:,}] --- ✨")
                    print(f"  • Total Artworks Vaulted:  {total_matched:,}")
                    print(f"  • Unique Artists Ingested: {len(unique_artists):,}")
                    if unique_artists:
                        print(f"  • Sample Active Artists:   {', '.join(list(unique_artists)[-5:])}")
                    print("─" * 60)

            # Flush remaining elements sitting inside trailing accumulators
            if accumulator:
                leftover_chunks = [accumulator[i:i + chunk_size] for i in range(0, len(accumulator), chunk_size)]
                parallel_outputs = pool.map(_extract_features_worker, leftover_chunks)
                for core_output in parallel_outputs:
                    for item in core_output:
                        start_row = master_index.ntotal
                        master_index.add(item['descriptors'])
                        
                        filename = f"{item['visual_id']}.jpg"
                        blob = self.state.bucket.blob(f"images/{filename}")
                        blob.upload_from_string(item['image_bytes'], content_type='image/jpeg')
                        
                        cache.append({
                            'id': item['visual_id'],
                            'title': str(item['title']),
                            'artist': str(item['artist']),
                            'url': item['url'],
                            'start_row': start_row,
                            'end_row': master_index.ntotal - 1
                            
                        })
                pbar.update(len(accumulator))

        if cache:
            vault_checkpoint(self.state, cache, master_index)
            print(f"\n💾 Flushing final checkpoint slice to GCS. Index length: {master_index.ntotal:,}")
        
        print("🏁 Parallel execution pool closed down safely.")
