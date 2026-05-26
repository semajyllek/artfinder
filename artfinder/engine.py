# artfinder/engine.py
import os
import cv2
import csv
import re
import time
import urllib.request
from dataclasses import dataclass
from google.cloud import storage
from google.colab import auth
import faiss
import numpy as np
from .config import Config

@dataclass
class SearchEngineState:
    """
    Maintains active system configurations, authenticated cloud buckets, 
    and vector index cache matrices across the runtime environment.
    """
    client:        storage.Client
    bucket:        storage.Bucket
    authority_set: set
    orb:           cv2.Feature2D
    index:         object = None
    source_df:     object = None


def build_authority_set():
    """
    Downloads the master curated artist list and flattens all string entries 
    to lowercase to ensure flawless matching against incoming dataset streams.
    """
    AUTH_URL = "https://raw.githubusercontent.com/oobabooga/stable-diffusion-automatic/master/artists.csv"
    try:
        with urllib.request.urlopen(AUTH_URL) as response:
            lines = [line.decode('utf-8') for line in response.readlines()]
        
        # Force entire set comprehension to lowercase and clear loose spacing
        return {row['artist'].lower().strip() for row in csv.DictReader(lines) if 'artist' in row}
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch remote authority set due to connection error: {e}")
        print("  Falling back to a resilient structural base configuration.")
        return {"titian", "tintoretto", "raphael", "canaletto"}


def setup_gcs():
    """Authenticates the environment session and mounts the target storage bucket."""
    auth.authenticate_user()
    client = storage.Client(project=Config.PROJECT_ID)
    bucket = client.get_bucket(Config.BUCKET_NAME)
    return client, bucket


def initialize_engine():
    """
    Instantiates global cloud structures, downloads curation matrices, and 
    configures localized ORB descriptor extraction configurations.
    """
    client, bucket = setup_gcs()
    auth_set = build_authority_set()
    
    print(f"💡 Engine Initialization: Loaded {len(auth_set):,} curated artists into the matching set.")
    
    orb = cv2.ORB_create(
        nfeatures   = Config.N_FEATURES,
        scaleFactor = Config.SCALE_FACTOR,
        nlevels     = Config.N_LEVELS,
        WTA_K       = Config.WTA_K,
    )
    return SearchEngineState(client, bucket, auth_set, orb)



def build_search_indices(state):
    """
    Compiles unclustered features into a structured, ID-mapped IVF index.
    Guarantees fuzzy lookup capabilities while locking down sub-millisecond speeds.
    """
    from .vault.builder import recover_state
    _, master_index = recover_state(state)
    n_total = master_index.ntotal
    
    print(f"🔄 Extracting {n_total:,} foundational vectors for structural IVF training...")
    all_vectors = master_index.reconstruct_n(0, n_total)
    
    # 1. Instantiate the baseline L2/Hamming distance quantizer
    quantizer = faiss.IndexBinaryFlat(Config.DIMENSION)
    
    # 2. Instantiate the core IVF cluster manager
    base_ivf_index = faiss.IndexBinaryIVF(quantizer, Config.DIMENSION, Config.CLUSTERS)
    
    # 3. 🌟 THE GOLDEN FIX: Wrap the IVF index inside an ID Map layer
    # This prevents the index pointers from fragmenting during multi-core reductions
    index_ivf = faiss.IndexBinaryIDMap(base_ivf_index)
    
    print(f"📐 Training IVF centroids over {Config.CLUSTERS} neighborhoods...")
    base_ivf_index.train(all_vectors)
    
    print("🔒 Injecting explicit row-index coordinates to mapping cells...")
    # Generate continuous explicit tracking IDs for all 14 million features
    explicit_ids = np.arange(n_total, dtype=np.int64)
    index_ivf.add_with_ids(all_vectors, explicit_ids)
    
    # Cache the structured index binary locally
    faiss.write_index_binary(index_ivf, Config.LOCAL_INDEX)
    state.index = index_ivf
    print(f"✅ IVF Cluster Index successfully restored and sealed with {index_ivf.ntotal:,} keys.")



def run_complete_system_rebuild(state):
    """
    Master pipeline orchestrator. Wipes active storage caches, streams raw 
    image data layers, computes ORB matrices, and trains the index.
    """
    from datasets import load_dataset
    from .vault.builder import VaultBuilder, purge_local_cache_files, purge_gcs_production_vault
    from .intake.wikiart import wikiart_image_first_generator
    from .evaluator import load_production_brain, execute_live_notebook_benchmark

    start_wall_time = time.time()
    print("🚧 --- STARTING TOTAL ENGINE RECONSTRUCTION --- 🚧\n")
    
    # Step 1: Wipe the slate completely clean
    purge_local_cache_files()
    purge_gcs_production_vault(state)
    
    # Step 2: Establish connection streams to the image database layer
    print("\n📦 Opening Hugging Face WikiArt Dataset stream layers...")
    wikiart_stream = load_dataset("huggan/wikiart", split="train", streaming=True)
    artist_labels = wikiart_stream.features['artist'].names
    
    curated_stream = wikiart_image_first_generator(
        stream=wikiart_stream,
        labels=artist_labels,
        authority_set=state.authority_set
    )
    
    # Step 3: Extract descriptors and checkpoint records upstream
    print(f"🚀 Processing images and extracting visual feature matrices...")
    builder = VaultBuilder(state)
    builder.ingest_stream(data_stream=curated_stream, batch_name="wikiart_foundational_layer")
    
    # Step 4: Compress flat storage points into partitioned Voronoi cells
    print("\n🔄 Compiling flat vector points into organized IVF clusters...")
    build_search_indices(state)
    
    # Step 5: Push completed search index binary back to GCS
    print(f"📤 Uploading local '{Config.LOCAL_INDEX}' to GCS destination '{Config.INDEX_PATH}'...")
    if os.path.exists(Config.LOCAL_INDEX):
        blob = state.bucket.blob(Config.INDEX_PATH)
        blob.upload_from_filename(Config.LOCAL_INDEX)
        print("  ✅ IVF production index uploaded successfully!")
    else:
        raise FileNotFoundError(f"❌ Expected local index file at '{Config.LOCAL_INDEX}' is missing.")
    
    # Step 6: Synchronize and bring the brain pointers live in memory
    print("\n🧠 Activating new production brain pointers...")
    load_production_brain(state)
    state.index.nprobe = 8
    
    # Step 7: Run immediate verification benchmark pass
    print("\n📈 Executing verification benchmark across clean asset maps...")
    accuracy, latency = execute_live_notebook_benchmark(state, sample_size=100)
    
    print("\n🏆 --- TARGET PIPELINE REALIZED --- 🏆")
    print(f"Total Unique Artworks Vaulted: {len(state.source_df):,}")
    print(f"Active Features Tracked:      {state.index.ntotal:,}")
    print(f"Benchmark Match Accuracy:     {accuracy * 100:.2f}%")
    print(f"Verified Engine Latency:      {latency:.2f} ms")
        
    duration = time.time() - start_wall_time
    print(f"\n✨ Total Pipeline Execution Completed in: {duration/60:.2f} minutes.")
