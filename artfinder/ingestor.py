import os
import gc
import json
import faiss
import numpy as np
import pandas as pd
import urllib.request
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from .config import Config
from .engine import is_curated_artist

def load_source_metadata(bucket):
    blob = bucket.blob(Config.META_PATH)
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_META)
        return pd.read_parquet(Config.LOCAL_META)
    return pd.DataFrame(columns=['id', 'title', 'artist', 'url', 'start_row', 'end_row'])

def recover_state(state):
    source_df = load_source_metadata(state.bucket)
    blob = state.bucket.blob(Config.VAULT_PATH)
    if blob.exists():
        blob.download_to_filename(Config.LOCAL_VAULT)
        master_index = faiss.read_index_binary(Config.LOCAL_VAULT)
    else:
        master_index = faiss.IndexBinaryFlat(Config.DIMENSION)
    return source_df, master_index

def resolve_image_url(row):
    for col in ['ImageURL', 'URL', 'ThumbnailURL']:
        if col in row and str(row[col]) != 'nan':
            url = str(row[col]).strip()
            if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']) or "media.moma.org" in url:
                return url
    return None

def onboard_artwork(row, master_index, state):
    obj_id  = str(row['ObjectID'])
    img_url = resolve_image_url(row)
    if not img_url: return None
    try:
        req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=Config.TIMEOUT) as resp:
            if "image" not in resp.info().get_content_type(): return None
            content = resp.read()
        img = Image.open(BytesIO(content)).convert('L')
        img.thumbnail(Config.RESIZE_DIM)
        kp, des = state.orb.detectAndCompute(np.array(img), None)
        if des is not None:
            start_row = master_index.ntotal
            master_index.add(des)
            state.bucket.blob(f"images/moma_{obj_id}.jpg").upload_from_string(content, content_type='image/jpeg')
            return {'id': obj_id, 'title': str(row['Title']), 'artist': str(row['Artist']), 'url': f"https://moma.org/works/{obj_id}", 'start_row': start_row, 'end_row': master_index.ntotal - 1}
    except: return None
    return None



def get_vault_stats(state):
    """Calculates total vectors currently in the binary vault."""
    _, master_index = recover_state(state)
    print(f"total vectors in vault: {master_index.ntotal:,}")
    return master_index.ntotal

def get_index_density(state):
    """Reports the average number of ORB features per artwork."""
    source_df      = load_source_metadata(state.bucket)
    total_paintings = len(source_df)

    _, master_index = recover_state(state)
    total_vectors   = master_index.ntotal

    if total_paintings == 0:
        return 0

    avg_features = total_vectors / total_paintings
    print(f"--- Index Density Report ---")
    print(f"total paintings: {total_paintings:,}")
    print(f"total vectors:   {total_vectors:,}")
    print(f"avg features:    {avg_features:.2f} per painting")

    return avg_features





def run_sync_cycle(state):
    source_df, master_index = recover_state(state)
    potential = state.df_moma[state.df_moma['ImageURL'].str.contains(r'\.jpg|\.jpeg|\.png|media\.moma\.org', case=False, na=False)].copy()
    universe = potential[potential['Artist'].apply(lambda x: is_curated_artist(x, state.authority_set))]
    known_ids = set(source_df['id'].astype(str))
    delta = universe[~universe['ObjectID'].astype(str).isin(known_ids)].head(Config.BATCH_LIMIT)
    
    cache = []
    for _, row in tqdm(delta.iterrows(), total=len(delta), desc="onboarding"):
        record = onboard_artwork(row, master_index, state)
        if record: cache.append(record)
        if len(cache) >= Config.CHECKPOINT_SIZE:
            # Checkpointing logic from your notebook
            updated_source = pd.concat([load_source_metadata(state.bucket), pd.DataFrame(cache)], ignore_index=True)
            updated_source.to_parquet(Config.LOCAL_META, index=False)
            state.bucket.blob(Config.META_PATH).upload_from_filename(Config.LOCAL_META)
            faiss.write_index_binary(master_index, Config.LOCAL_VAULT)
            state.bucket.blob(Config.VAULT_PATH).upload_from_filename(Config.LOCAL_VAULT)
            cache = [] ; gc.collect()
