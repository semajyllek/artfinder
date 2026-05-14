import cv2
import pandas as pd
import re
import csv
import urllib.request
from dataclasses import dataclass
from google.cloud import storage
from google.colab import auth
from fuzzywuzzy import fuzz
from .config import Config
import faiss
import numpy as np


@dataclass
class SearchEngineState:
    client:        storage.Client
    bucket:        storage.Bucket
    df_moma:       pd.DataFrame
    authority_set: set
    orb:           cv2.Feature2D
    index:     object = None
    source_df: object = None
    id_map:    object = None



def build_search_indices(state):
    """
    Performs K-means clustering on vaulted vectors to create an IVF index.
    Essential for scaling toward the 50,000 image limit.
    """
    # 1. Recover the raw vectors from the vault
    from .ingestor import recover_state
    _, master_index = recover_state(state)
    
    # Extract the underlying data as a numpy array
    # IndexBinaryFlat stores data in a way that allows reconstruction
    n_total = master_index.ntotal
    print(f"Reconstructing {n_total:,} vectors for training...")
    
    # Get all vectors (MoMA + Met)
    all_vectors = master_index.reconstruct_n(0, n_total)
    
    # 2. Configure the IVF Index
    # We use 4096 centroids to maintain high speed as we hit 50k images
    quantizer = faiss.IndexBinaryFlat(Config.DIMENSION)
    index_ivf = faiss.IndexBinaryIVF(quantizer, Config.DIMENSION, 4096)
    
    # 3. Train the Index
    # This is the 'Brain Training'—it groups similar visual features together
    print(f"Training IVF Index with 4096 centroids...")
    index_ivf.train(all_vectors)
    
    # 4. Add the vectors to the new structure
    print("Populating IVF Index...")
    index_ivf.add(all_vectors)
    
    # 5. Save locally
    faiss.write_index_binary(index_ivf, Config.LOCAL_INDEX)
    state.index = index_ivf
    print(f"✅ Search Index rebuilt successfully with {index_ivf.ntotal:,} vectors.")


def setup_gcs():
    auth.authenticate_user()
    client = storage.Client(project=Config.PROJECT_ID)
    bucket = client.get_bucket(Config.BUCKET_NAME)
    return client, bucket

def load_moma_universe():
    MOMA_URL = "https://github.com/MuseumofModernArt/collection/raw/main/Artworks.csv?download=true"
    return pd.read_csv(MOMA_URL, low_memory=False, on_bad_lines='skip')

def build_authority_set():
    AUTH_URL = "https://raw.githubusercontent.com/oobabooga/stable-diffusion-automatic/master/artists.csv"
    with urllib.request.urlopen(AUTH_URL) as response:
        lines = [line.decode('utf-8') for line in response.readlines()]
    return {row['artist'].lower().strip() for row in csv.DictReader(lines)}


def is_curated_artist(name, authority_set):
    """
    Determines if an artist exists in the curated set using 
    cleaning and fuzzy matching.
    """
    if not name or str(name).lower() in ['unknown', 'unidentified artist', '']:
        return False
    
    # 1. CLEANING: Lowercase and strip parenthetical "noise" (dates, locations)
    # This turns "Rembrandt (Rembrandt van Rijn)" -> "rembrandt"
    clean_name = str(name).lower()
    clean_name = re.sub(r'\(.*\)', '', clean_name).strip()
    
    # 2. EXACT MATCH: Fast check against the authority set
    if clean_name in authority_set:
        return True
        
    # 3. FUZZY MATCH: Token-based comparison to handle name order
    # token_set_ratio handles "Sargent, John" vs "John Sargent" perfectly
    # We use a threshold of 90 to maintain precision.
    for curated_name in authority_set:
        # Avoid checking very short strings to prevent false positives
        if len(clean_name) < 4: continue 
        
        ratio = fuzz.token_set_ratio(clean_name, curated_name)
        if ratio >= 90:
            return True
            
    return False



def initialize_engine():
    client, bucket = setup_gcs()
    df_moma   = load_moma_universe()
    auth_set  = build_authority_set()
    orb       = cv2.ORB_create(
                    nfeatures   = Config.N_FEATURES,
                    scaleFactor = Config.SCALE_FACTOR,
                    nlevels     = Config.N_LEVELS,
                    WTA_K       = Config.WTA_K,
                )
    return SearchEngineState(client, bucket, df_moma, auth_set, orb)
