import os
import gc
import json
import faiss
import requests
import numpy as np
import pandas as pd
import urllib.request
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from .config import Config
from .engine import is_curated_artist

class BaseIngestor(ABC):
    def __init__(self, state):
        self.state = state

    @abstractmethod
    def fetch_delta(self, known_ids, limit):
        """Source-specific logic to find new artworks."""
        pass

    def process_and_vault(self, delta, master_index):
        """Standardized ORB extraction and GCS vaulting logic."""
        cache = []
        for _, row in tqdm(delta.iterrows(), total=len(delta), desc=f"Onboarding"):
            record = self._onboard_artwork(row, master_index)
            if record: 
                cache.append(record)
            
            if len(cache) >= Config.CHECKPOINT_SIZE:
                self._vault_checkpoint(cache, master_index)
                cache = []
                gc.collect()
        
        if cache:
            self._vault_checkpoint(cache, master_index)

    def _onboard_artwork(self, row, master_index):
        obj_id = str(row['ObjectID'])
        img_url = row['ImageURL']
        try:
            req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=Config.TIMEOUT) as resp:
                content = resp.read()
            
            img = Image.open(BytesIO(content)).convert('L')
            img.thumbnail(Config.RESIZE_DIM)
            kp, des = self.state.orb.detectAndCompute(np.array(img), None)

            if des is not None:
                start_row = master_index.ntotal
                master_index.add(des)
                # Note: prefixing filenames by source is good practice
                self.state.bucket.blob(f"images/{row['Source']}_{obj_id}.jpg").upload_from_string(content, content_type='image/jpeg')
                return {
                    'id': obj_id, 'title': str(row['Title']), 'artist': str(row['Artist']),
                    'url': row['SourceURL'], 'start_row': start_row, 'end_row': master_index.ntotal - 1
                }
        except: return None

    def _vault_checkpoint(self, new_records, master_index):
        # Implementation matches your original checkpointing logic
        current_source = load_source_metadata(self.state.bucket)
        updated_source = pd.concat([current_source, pd.DataFrame(new_records)], ignore_index=True)
        updated_source.to_parquet(Config.LOCAL_META, index=False)
        self.state.bucket.blob(Config.META_PATH).upload_from_filename(Config.LOCAL_META)
        faiss.write_index_binary(master_index, Config.LOCAL_VAULT)
        self.state.bucket.blob(Config.VAULT_PATH).upload_from_filename(Config.LOCAL_VAULT)

# --- Helper Functions (Same as Notebook) ---
def load_source_metadata(bucket):
    blob = bucket.blob(Config.META_PATH)
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_META)
        return pd.read_parquet(Config.LOCAL_META)
    return pd.DataFrame(columns=['id', 'title', 'artist', 'url', 'start_row', 'end_row'])

def recover_state(state):
    source_df = load_source_metadata(state.bucket)
    blob = state.bucket.blob(Config.VAULT_PATH)
    master_index = faiss.read_index_binary(Config.LOCAL_VAULT) if blob.exists() else faiss.IndexBinaryFlat(Config.DIMENSION)
    return source_df, master_index
