import cv2
import os
import pandas as pd
import re
import csv
import urllib.request
from dataclasses import dataclass
from google.cloud import storage
from google.colab import auth
import faiss
import numpy as np
from .config import Config

@dataclass
class SearchEngineState:
    client:        storage.Client
    bucket:        storage.Bucket
    authority_set: set
    orb:           cv2.Feature2D
    index:     object = None
    source_df: object = None

def build_search_indices(state):
    """Performs K-means clustering on vaulted vectors to create an IVF index."""
    from .vault.builder import recover_state
    _, master_index = recover_state(state)
    n_total = master_index.ntotal
    print(f"Reconstructing {n_total:,} vectors for training...")
    all_vectors = master_index.reconstruct_n(0, n_total)
    
    quantizer = faiss.IndexBinaryFlat(Config.DIMENSION)
    index_ivf = faiss.IndexBinaryIVF(quantizer, Config.DIMENSION, Config.CLUSTERS)
    
    print(f"Training IVF Index with {Config.CLUSTERS} centroids...")
    index_ivf.train(all_vectors)
    index_ivf.add(all_vectors)
    
    faiss.write_index_binary(index_ivf, Config.LOCAL_INDEX)
    state.index = index_ivf
    print(f"✅ Search Index rebuilt successfully with {index_ivf.ntotal:,} vectors.")

def setup_gcs():
    auth.authenticate_user()
    client = storage.Client(project=Config.PROJECT_ID)
    bucket = client.get_bucket(Config.BUCKET_NAME)
    return client, bucket

def build_authority_set():
    AUTH_URL = "https://raw.githubusercontent.com/oobabooga/stable-diffusion-automatic/master/artists.csv"
    with urllib.request.urlopen(AUTH_URL) as response:
        lines = [line.decode('utf-8') for line in response.readlines()]
    return {row['artist'].lower().strip() for row in csv.DictReader(lines)}

def initialize_engine():
    client, bucket = setup_gcs()
    auth_set  = build_authority_set()
    orb       = cv2.ORB_create(
                    nfeatures   = Config.N_FEATURES,
                    scaleFactor = Config.SCALE_FACTOR,
                    nlevels     = Config.N_LEVELS,
                    WTA_K       = Config.WTA_K,
                )
    return SearchEngineState(client, bucket, auth_set, orb)


def run_complete_system_rebuild(state):
    """Orchestrates the entire image-first rebuild pipeline entirely within functions."""
    import time
    from datasets import load_dataset
    from .vault.builder import VaultBuilder, purge_local_cache_files, purge_gcs_production_vault
    from .intake.wikiart import wikiart_image_first_generator
    from .evaluator import load_production_brain, execute_live_notebook_benchmark

    start_wall_time = time.time()
    print("🚧 --- STARTING TOTAL ENGINE RECONSTRUCTION --- 🚧\n")
    
    purge_local_cache_files()
    purge_gcs_production_vault(state)
    
    print("\n📦 Opening Hugging Face WikiArt Dataset stream layers...")
    wikiart_stream = load_dataset("huggan/wikiart", split="train", streaming=True)
    artist_labels = wikiart_stream.features['artist'].names
    
    curated_stream = wikiart_image_first_generator(
        stream=wikiart_stream,
        labels=artist_labels,
        authority_set=state.authority_set
    )
    
    print(f"🚀 Extracting ORB features for {len(state.authority_set)} target artists...")
    builder = VaultBuilder(state)
    builder.ingest_stream(data_stream=curated_stream, batch_name="wikiart_foundational_layer")
    
    build_search_indices(state)
    
    print(f"📤 Uploading local '{Config.LOCAL_INDEX}' to GCS destination '{Config.INDEX_PATH}'...")
    if os.path.exists(Config.LOCAL_INDEX):
        blob = state.bucket.blob(Config.INDEX_PATH)
        blob.upload_from_filename(Config.LOCAL_INDEX)
        print("  ✅ IVF production index uploaded successfully!")
    else:
        raise FileNotFoundError(f"❌ Expected local index file at '{Config.LOCAL_INDEX}' is missing.")
    
    print("\n🧠 Activating new production brain pointers...")
    load_production_brain(state)
    state.index.nprobe = 8
    
    print("\n📈 Executing verification benchmark across clean asset maps...")
    accuracy, latency = execute_live_notebook_benchmark(state, sample_size=100)
    
    print("\n🏆 --- TARGET PIPELINE REALIZED --- 🏆")
    print(f"Total Unique Artworks Vaulted: {len(state.source_df):,}")
    print(f"Active Features Tracked:      {state.index.ntotal:,}")
    print(f"Benchmark Match Accuracy:     {accuracy * 100:.2f}%")
    print(f"Verified Engine Latency:      {latency:.2f} ms")
        
    duration = time.time() - start_wall_time
    print(f"\n✨ Total Pipeline Execution Completed in: {duration/60:.2f} minutes.")
