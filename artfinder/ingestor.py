import os
import gc
import faiss
import numpy as np
import pandas as pd
import urllib.request
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from abc import ABC, abstractmethod
from .config import Config

# ─── BASE CLASS ──────────────────────────────────────────────────────────────

class BaseIngestor(ABC):
    """
    Abstract Base Class to ensure all museum sources use the same
    optimized ORB and resizing parameters.
    """
    def __init__(self, state):
        self.state = state

    @abstractmethod
    def fetch_delta(self, known_ids, limit):
        """Must return a DataFrame with: ObjectID, Title, Artist, ImageURL, SourceURL, Source."""
        pass

    def process_and_vault(self, delta, master_index):
        """Standardized extraction and vaulting loop."""
        if delta.empty:
            print(f"System fully synced for this source.")
            return

        cache = []
        source_name = delta['Source'].iloc[0] if 'Source' in delta.columns else "Unknown"
        
        for _, row in tqdm(delta.iterrows(), total=len(delta), desc=f"Syncing {source_name}"):
            record = onboard_artwork(row, master_index, self.state)
            if record: 
                cache.append(record)
            
            if len(cache) >= Config.CHECKPOINT_SIZE:
                vault_checkpoint(self.state, cache, master_index)
                cache = []
                gc.collect()
        
        if cache:
            vault_checkpoint(self.state, cache, master_index)

# ─── CORE VAULT LOGIC ────────────────────────────────────────────────────────

def load_source_metadata(bucket):
    """Downloads source_metadata.parquet from GCS."""
    blob = bucket.blob(Config.META_PATH)
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_META)
        return pd.read_parquet(Config.LOCAL_META)
    return pd.DataFrame(columns=['id', 'title', 'artist', 'url', 'start_row', 'end_row'])

def recover_state(state):
    """
    Recovers metadata and the binary vector vault from GCS.
    Automatically initializes the native integer ID mapping layout array.
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
    
    # Sync and type the memory map mapping array on state restoration
    sync_runtime_integer_map(state, master_index)
    
    return source_df, master_index

def vault_checkpoint(state, new_records, master_index):
    """
    Saves progress to GCS to prevent data loss during long syncs.
    Automatically regenerates and synchronizes the active runtime integer map.
    """
    if not new_records: return
    current_source = load_source_metadata(state.bucket)
    updated_source = pd.concat([current_source, pd.DataFrame(new_records)], ignore_index=True)

    updated_source.to_parquet(Config.LOCAL_META, index=False)
    state.bucket.blob(Config.META_PATH).upload_from_filename(Config.LOCAL_META)

    faiss.write_index_binary(master_index, Config.LOCAL_VAULT)
    state.bucket.blob(Config.VAULT_PATH).upload_from_filename(Config.LOCAL_VAULT)
    
    # Enforce active context updates to synchronize the dynamic state frame fields
    state.source_df = updated_source
    state.index = master_index
    
    sync_runtime_integer_map(state, master_index)

def sync_runtime_integer_map(state, master_index=None):
    """
    High-speed mapping utility function. Converts alphanumeric string unique tracking keys 
    into contiguous native standard integer indices matching DataFrame sequence rows.
    Prevents runtime casting exceptions when running vector vote-tally calculations.
    """
    if state.source_df is None or state.source_df.empty:
        return

    active_index = master_index if master_index is not None else state.index
    if active_index is None:
        return

    max_row = int(state.source_df['end_row'].max()) if not state.source_df['end_row'].isna().all() else 0
    if max_row == 0 and active_index.ntotal == 0:
        state.id_map = np.zeros(0, dtype=np.int64)
        return

    # Build direct map: String Asset ID -> DataFrame Integer Row Index 
    id_to_row_idx = {str(uid): i for i, uid in enumerate(state.source_df['id'])}
    
    # Initialize native 64-bit integer tracking continuous vector lane slots
    id_map_ints = np.zeros(max_row + 1, dtype=np.int64)

    for _, row in state.source_df.iterrows():
        if pd.isna(row['start_row']) or pd.isna(row['end_row']):
            continue
        
        # Pull matching unique matrix registration location coordinates
        current_row_index = id_to_row_idx[str(row['id'])]
        
        # Populate whole feature vector span block directly with row key reference numbers
        id_map_ints[int(row['start_row']):int(row['end_row']) + 1] = current_row_index

    # Bind elements straight back into core state attributes
    state.id_map = id_map_ints
    state.index = active_index


# ─── PIPELINE ATTACHMENT LOGIC ───────────────────────────────────────────────

def resolve_image_url(row):
    """Identifies direct image assets from source rows."""
    for col in ['ImageURL', 'URL', 'ThumbnailURL']:
        if col in row and str(row[col]) != 'nan':
            url = str(row[col]).strip()
            if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']) or "media.moma.org" in url:
                return url
    return None

def onboard_artwork(row, master_index, state):
    """
    Processes an image, extracts ORB features, and uploads to GCS.
    Updated to handle GUIDs (met_123) without double-prefixing.
    """
    obj_id = str(row['ObjectID'])
    img_url = resolve_image_url(row)
    source_label = row.get('Source', 'moma')

    if not img_url: 
        return None

    try:
        # 1. Fetch Image
        req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=Config.TIMEOUT) as resp:
            content = resp.read()

        # 2. Process for ORB
        img = Image.open(BytesIO(content)).convert('L')
        img.thumbnail(Config.RESIZE_DIM)
        kp, des = state.orb.detectAndCompute(np.array(img), None)

        if des is not None:
            start_row = master_index.ntotal
            master_index.add(des)

            # 3. FIXED PATH LOGIC: Prevent double-prefixing
            filename = f"{obj_id}.jpg" if "_" in obj_id else f"{source_label}_{obj_id}.jpg"
            blob = state.bucket.blob(f"images/{filename}")
            
            blob.upload_from_string(content, content_type='image/jpeg')

            return {
                'id': obj_id, 
                'title': str(row['Title']), 
                'artist': str(row['Artist']),
                'url': row.get('SourceURL', img_url), # Fallback to img_url if SourceURL missing
                'start_row': start_row, 
                'end_row': master_index.ntotal - 1
            }
    except Exception as e: 
        print(f"Error onboarding {obj_id}: {e}")
        return None
    return None


# ─── REPORTING HELPERS ───────────────────────────────────────────────────────

def get_index_density(state):
    """Calculates the feature-to-painting ratio."""
    source_df = load_source_metadata(state.bucket)
    total_paintings = len(source_df)

    _, master_index = recover_state(state)
    total_vectors = master_index.ntotal

    if total_paintings == 0:
        return 0

    avg_features = total_vectors / total_paintings
    print(f"--- Index Density Report ---")
    print(f"total paintings: {total_paintings:,}")
    print(f"total vectors:   {total_vectors:,}")
    print(f"avg features:    {avg_features:.2f} per painting")

    return avg_features
