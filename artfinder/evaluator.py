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
    """
    Runs retrieval evaluation on a specific list of IDs.
    Automatically synchronizes metadata and resolves image source paths.
    """
    import time
    from io import BytesIO
    from PIL import Image
    from tqdm.auto import tqdm
    from .ingestor import load_source_metadata  # Ensure metadata helper is accessible

    # 1. AUTO-REFRESH: Sync local state with GCS to include newly vaulted items
    state.source_df = load_source_metadata(state.bucket)
    
    image_results = []
    correct = 0
    run_id = time.strftime("%Y%m%d_%H%M%S")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # 2. ITERATION LOOP
    for obj_id in tqdm(test_ids, desc="Verifying Batch"):
        # Safe lookup: ensures the ID exists in the refreshed metadata
        mask = state.source_df['id'] == str(obj_id)
        if not mask.any():
            print(f"⚠️ Warning: ID {obj_id} not found in metadata. Skipping.")
            continue
            
        row = state.source_df[mask].iloc[0]
        
        # 3. DYNAMIC PREFIXING: Resolve folder path based on URL patterns
        url_str = str(row['url']).lower()
        if "metmuseum.org" in url_str:
            source_prefix = "met"
        elif "artic.edu" in url_str:
            source_prefix = "aic"
        else:
            source_prefix = "moma"
            
        # 4. IMAGE RETRIEVAL & SIMULATION
        try:
            blob_img = state.bucket.blob(f"images/{source_prefix}_{obj_id}.jpg")
            raw_img = np.array(Image.open(BytesIO(blob_img.download_as_bytes())).convert("RGB"))
            
            # Apply the handheld gallery simulation (affine transform, background, etc.)
            test_photo = apply_simulation(raw_img)
            
            t0 = time.perf_counter()
            # Perform retrieval using the optimized ORB/FAISS pipeline
            meta, conf = identify_art(state, test_photo, nprobe=nprobe)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)

            is_match = (str(meta["id"]) == str(obj_id)) if meta else False
            if is_match:
                correct += 1

            # 5. LOGGING
            image_results.append({
                "run_id": run_id,
                "timestamp": ts,
                "n_features": Config.N_FEATURES,
                "nprobe": nprobe,
                "true_id": obj_id,
                "predicted_id": meta["id"] if meta else None,
                "predicted_title": meta["title"] if meta else None,
                "predicted_artist": meta["artist"] if meta else None,
                "confidence": round(conf, 4) if meta else 0.0,
                "match": is_match,
                "latency_ms": latency_ms,
            })
            
            # Optional visual proof for failures or if not silent
            if not silent or not is_match:
                from .evaluator import show_3panel
                show_3panel(state, test_photo, meta, conf, obj_id)
                
        except Exception as e:
            print(f"❌ Error processing ID {obj_id}: {e}")
            continue

    # 6. SUMMARY CALCULATION
    n_tested = len(image_results)
    accuracy = correct / n_tested if n_tested > 0 else 0
    avg_latency = round(sum(r["latency_ms"] for r in image_results) / n_tested, 2) if n_tested > 0 else 0
    
    summary_row = {
        "run_id": run_id,
        "timestamp": ts,
        "n_features": Config.N_FEATURES,
        "vault_size": len(state.source_df),
        "nprobe": nprobe,
        "n_tested": n_tested,
        "n_correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": avg_latency,
    }

    return image_results, summary_row



def show_3panel(state, test_photo, meta, confidence, true_id):
    """
    Generates a visual verification plot comparing the simulated query 
    to the vault's best match.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    from io import BytesIO

    # 1. Resolve True Identity Image
    true_row = state.source_df[state.source_df['id'] == str(true_id)].iloc[0]
    true_url = str(true_row['url']).lower()
    prefix = "met" if "metmuseum.org" in true_url else "moma" # Add "aic" if needed
    
    true_blob = state.bucket.blob(f"images/{prefix}_{true_id}.jpg")
    true_img = Image.open(BytesIO(true_blob.download_as_bytes()))

    # 2. Resolve Predicted Image (if any)
    pred_img = None
    if meta:
        pred_row = state.source_df[state.source_df['id'] == str(meta['id'])].iloc[0]
        pred_url = str(pred_row['url']).lower()
        p_prefix = "met" if "metmuseum.org" in pred_url else "moma"
        pred_blob = state.bucket.blob(f"images/{p_prefix}_{meta['id']}.jpg")
        pred_img = Image.open(BytesIO(pred_blob.download_as_bytes()))

    # 3. Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(test_photo)
    ax[0].set_title(f"Query (Simulated)\nTrue ID: {true_id}")
    ax[0].axis('off')

    ax[1].imshow(true_img)
    ax[1].set_title(f"Vault Reference\n{true_row['title'][:30]}...")
    ax[1].axis('off')

    if pred_img:
        ax[2].imshow(pred_img)
        match_status = "✅ MATCH" if str(meta['id']) == str(true_id) else "❌ MISMATCH"
        ax[2].set_title(f"{match_status}\nConf: {confidence:.2f}")
    else:
        ax[2].set_title("No Match Found")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()
