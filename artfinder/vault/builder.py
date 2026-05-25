import os
import gc
import cv2
import faiss
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
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
        # Initialize an empty, flat uncompressed array if no cloud cache exists
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
    """Deletes existing binary vaults and parquets inside the active cloud bucket."""
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


# ──────────────────────────────────────────────────────────────────────────────
# 3. UNIFIED INGESTION BUILDER ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class VaultBuilder:
    """
    Modular execution engine for building the core visual database.
    Decoupled into abstract intake parsing streams and a unified feature vault loop.
    """
    def __init__(self, state):
        self.state = state

    def ingest_stream(self, data_stream, batch_name, total_records=None):
        """
        Unified ingestion gateway with explicit outer-loop telemetry tracking
        to prevent tqdm progress bars from swallowing the dashboard output.
        """
        _, master_index = recover_state(self.state)
        
        try:
            current_df = load_source_metadata(self.state.bucket)
            known_ids = set(current_df['id'].dropna().unique())
        except Exception:
            known_ids = set()

        cache = []
        
        # Outer Loop Telemetry Real-time Trackers
        total_scanned = 0
        total_matched = 0
        unique_artists = set()

        print(f"🚀 Initializing unified ingestion loop for batch: {batch_name}")

        for record in tqdm(data_stream, desc=f"Vaulting {batch_name}", total=total_records):
            total_scanned += 1
            visual_id = record['visual_id']
            
            # Feed real-time uniqueness ledger trackers directly
            if 'artist' in record:
                unique_artists.add(record['artist'])

            if visual_id in known_ids:
                continue

            try:
                pil_img = record['image']
                if isinstance(pil_img, Image.Image):
                    img_np = np.array(pil_img.convert('RGB'))
                else:
                    continue

                # Standardize spatial dimension sizes
                resized = cv2.resize(img_np, Config.RESIZE_DIM)
                kp, des = self.state.orb.detectAndCompute(resized, None)

                if des is not None and len(des) > 0:
                    total_matched += 1
                    start_row = master_index.ntotal
                    master_index.add(des)

                    # Compress and pipeline query image representations to cloud targets
                    buffer = BytesIO()
                    pil_img.convert('RGB').save(buffer, format="JPEG", quality=85)
                    content = buffer.getvalue()

                    filename = f"{visual_id}.jpg"
                    blob = self.state.bucket.blob(f"images/{filename}")
                    blob.upload_from_string(content, content_type='image/jpeg')

                    cache.append({
                        'id': visual_id,
                        'title': str(record['title']),
                        'artist': str(record['artist']),
                        'url': record.get('SourceURL', 'https://www.wikiart.org'),
                        'start_row': start_row,
                        'end_row': master_index.ntotal - 1
                    })

                # Safely intercept outer line states to inject readable stream diagnostics
                if total_scanned % 1000 == 0:
                    print(f"\n✨ --- LIVE INGESTION DASHBOARD [Records Processed: {total_scanned:,}] --- ✨")
                    print(f"  • Total Artworks Accepted: {total_matched:,}")
                    print(f"  • Unique Artists Ingested: {len(unique_artists):,}")
                    if unique_artists:
                        print(f"  • Sample Active Artists:   {', '.join(list(unique_artists)[-5:])}")
                    print("─" * 60)

                if len(cache) >= Config.CHECKPOINT_SIZE:
                    vault_checkpoint(self.state, cache, master_index)
                    print(f"\n💾 Flushing safe checkpoint slice to GCS. Index length: {master_index.ntotal:,}")
                    cache = []
                    gc.collect()

            except Exception as e:
                print(f"Error onboarding asset {visual_id}: {e}")
                continue

        if cache:
            vault_checkpoint(self.state, cache, master_index)
            print(f"\n💾 Flushing final checkpoint slice to GCS. Index length: {master_index.ntotal:,}")
