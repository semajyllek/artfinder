import os
import cv2
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from .config import Config

def load_production_brain(state):
    """Downloads index and metadata from GCS and builds the RAM mapping."""
    print("Downloading Production Brain from GCS...")
    state.bucket.blob(Config.INDEX_PATH).download_to_filename(Config.LOCAL_INDEX)
    import faiss
    state.index = faiss.read_index_binary(Config.LOCAL_INDEX)

    state.bucket.blob(Config.META_PATH).download_to_filename(Config.LOCAL_META)
    state.source_df = pd.read_parquet(Config.LOCAL_META)

    # Build RAM Map (Maps millions of vectors to image indices)
    state.id_map = np.zeros(state.index.ntotal, dtype='uint32')
    for idx, row in tqdm(state.source_df.iterrows(), total=len(state.source_df), desc="Mapping Vault"):
        start, end = int(row['start_row']), int(row['end_row'])
        state.id_map[start : end + 1] = idx
    print(f"✅ Alignment confirmed. {state.index.ntotal:,} vectors mapped to {len(state.source_df):,} images.")

def tally_votes_weighted(faiss_indices, query_kp, id_map, source_df_len):
    """Applies exponential weighting to keypoints based on center-proximity."""
    h, w = 1000, 1000
    cx, cy = w // 2, h // 2
    weights = []
    for kp in query_kp:
        dist = np.sqrt((kp.pt[0] - cx)**2 + (kp.pt[1] - cy)**2)
        weight = np.exp(-dist**2 / (2 * (250**2)))
        weights.append(weight)

    expanded_weights = np.repeat(weights, 5)
    valid_mask = faiss_indices >= 0
    image_indices = id_map[faiss_indices[valid_mask]]
    valid_weights = expanded_weights[valid_mask]

    weighted_counts = np.bincount(image_indices, weights=valid_weights, minlength=source_df_len)
    winner_idx  = np.argmax(weighted_counts)
    confidence  = weighted_counts[winner_idx] / (np.sum(valid_weights) + 1e-6)
    return winner_idx, confidence

def identify_art(state, img_rgb, nprobe=32):
    """Core retrieval function using ORB descriptors and FAISS search."""
    state.index.nprobe = nprobe
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kp_q, des_q = state.orb.detectAndCompute(gray, None)
    if des_q is None: return None, 0

    _, I = state.index.search(des_q, 5)
    winner_idx, confidence = tally_votes_weighted(I.flatten(), kp_q, state.id_map, len(state.source_df))
    res = state.source_df.iloc[winner_idx]

    return {
        "id":     res['id'],
        "title":  res['title'],
        "artist": res['artist']
    }, confidence

def apply_simulation(img_rgb):
    """Simulates gallery conditions with perspective shifts and wall backgrounds."""
    bg_colors = [(181, 171, 156), (245, 245, 220), (128, 128, 128), (255, 255, 255), (30, 30, 30)]
    canvas = np.full((1000, 1000, 3), random.choice(bg_colors), dtype=np.uint8)

    rows, cols, _ = img_rgb.shape
    shift = 0.15 * min(rows, cols)
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
    pts2 = np.float32([
        [random.uniform(0, shift), random.uniform(0, shift)],
        [cols - random.uniform(0, shift), random.uniform(0, shift)],
        [random.uniform(0, shift), rows - random.uniform(0, shift)]
    ])

    matrix = cv2.getAffineTransform(pts1, pts2)
    warped = cv2.warpAffine(img_rgb, matrix, (cols, rows))
    scale  = min(750/rows, 750/cols)
    warped = cv2.resize(warped, (int(cols*scale), int(rows*scale)))

    h, w, _ = warped.shape
    y_off, x_off = (1000 - h) // 2, (1000 - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = warped
    return canvas


def run_final_exam_and_log(state, nprobe=32, silent=True, test_ids=None):
    """
    Runs evaluation. If test_ids is provided, it performs a targeted 
    verification of those specific items.
    """
    if test_ids is None:
        # Fallback to the standard manifest on GCS
        blob = state.bucket.blob(Config.MANIFEST_PATH)
        test_ids = json.loads(blob.download_as_string())["test_queries"]
    
    image_results, summary_row = collect_eval_results(state, test_ids, nprobe=nprobe, silent=silent)
    
    print(f"\n--- VALIDATION SCORE: {summary_row['accuracy']*100:.1f}% | Latency: {summary_row['avg_latency_ms']}ms ---")
    return image_results, summary_row

def collect_eval_results(state, test_ids, nprobe=32, silent=True):
    """Logic updated to accept a direct list of IDs."""
    image_results, correct = [], 0
    run_id, ts = time.strftime("%Y%m%d_%H%M%S"), time.strftime("%Y-%m-%d %H:%M:%S")

    for obj_id in tqdm(test_ids, desc="Verifying Batch"):
        # Dynamic prefixing: Find the source label in metadata to locate the image
        # This handles 'met_' vs 'moma_' automatically
        row = state.source_df[state.source_df['id'] == str(obj_id)].iloc[0]
        source_prefix = "met" if "metmuseum.org" in row['url'] else "moma"
        
        blob_img = state.bucket.blob(f"images/{source_prefix}_{obj_id}.jpg")
        # ... rest of the simulation and identification logic ...
