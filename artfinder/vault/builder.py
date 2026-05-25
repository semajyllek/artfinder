# artfinder/vault/builder.py
import os
import gc
import urllib.request
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from ..config import Config
from ..ingestor import recover_state, vault_checkpoint, load_source_metadata

class VaultBuilder:
    """
    Modular execution engine for building the core visual database.
    Decoupled into abstract intake parsing streams and a unified feature vault loop.
    """
    def __init__(self, state):
        self.state = state

    def ingest_stream(self, data_stream, batch_name, total_records=None):
        """
        The central, unified ingestion gateway. Expects an iterable stream of 
        standardized dictionaries: {'visual_id', 'image', 'title', 'artist', 'filename'}
        """
        _, master_index = recover_state(self.state)
        
        try:
            current_df = load_source_metadata(self.state.bucket)
            known_ids = set(current_df['id'].dropna().unique())
        except Exception:
            known_ids = set()

        cache = []
        print(f"🚀 Initializing unified ingestion loop for batch: {batch_name}")

        for record in tqdm(data_stream, desc=f"Vaulting {batch_name}", total=total_records):
            visual_id = record['visual_id']
            if visual_id in known_ids:
                continue

            try:
                # Direct feature extraction out of the stream's raw PIL image layer
                pil_img = record['image']
                if isinstance(pil_img, Image.Image):
                    img_np = np.array(pil_img.convert('RGB'))
                else:
                    continue

                # Quantize sizing barriers
                resized = cv2.resize(img_np, Config.RESIZE_DIM)
                kp, des = self.state.orb.detectAndCompute(resized, None)

                if des is not None and len(des) > 0:
                    start_row = master_index.ntotal
                    master_index.add(des)

                    # Save the raw compressed JPEG matrix to GCS bucket for visual inspection loops
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
