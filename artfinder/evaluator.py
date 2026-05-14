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
    if test_ids is None:
        blob = state.bucket.blob(Config.MANIFEST_PATH)
        test_ids = json.loads(blob.download_as_string())["test_queries"]
    
    image_results, summary_row = collect_eval_results(state, test_ids, nprobe=nprobe, silent=silent)
    print(f"\n--- VALIDATION SCORE: {summary_row['accuracy']*100:.1f}% | Latency: {summary_row['avg_latency_ms']}ms ---")
    return image_results, summary_row



def collect_eval_results(state, test_ids, nprobe=32, silent=True):
    """
    Main orchestrator for batch verification. 
    Breaks the batch into discrete tasks: fetch, simulate, identify, and score.
    """
    from .ingestor import load_source_metadata
    state.source_df = load_source_metadata(state.bucket)
    
    results = []
    run_id = time.strftime("%Y%m%d_%H%M%S")
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for obj_id in tqdm(test_ids, desc="Verifying Batch"):
        # 1. Fetch and Prepare
        raw_img = self._fetch_ground_truth_image(state, obj_id)
        if raw_img is None:
            continue

        # 2. Simulate Gallery Conditions
        test_photo = apply_simulation(raw_img)
        
        # 3. Identify and Measure Latency
        prediction, confidence, latency = self._run_inference(state, test_photo, nprobe)

        # 4. Score the result
        result_entry = self._score_prediction(
            obj_id, prediction, confidence, latency, run_id, ts
        )
        results.append(result_entry)
        
        # 5. Visual Feedback (Optional)
        if not silent or not result_entry['match']:
            show_3panel(state, test_photo, prediction, confidence, str(obj_id))
                
    return results, self._summarize_run(results, state, nprobe, run_id, ts)

def _fetch_ground_truth_image(self, state, obj_id):
    """Handles the GUID logic to retrieve the correct image from GCS."""
    obj_id_str = str(obj_id)
    
    # Path Logic: Support both 'met_123' and legacy '123' (moma_123)
    filename = f"{obj_id_str}.jpg" if "_" in obj_id_str else f"moma_{obj_id_str}.jpg"
    blob_path = f"images/{filename}"
    
    try:
        blob = state.bucket.blob(blob_path)
        return np.array(Image.open(BytesIO(blob.download_as_bytes())).convert("RGB"))
    except Exception:
        print(f"❌ Missing Image: {blob_path}")
        return None

def _run_inference(self, state, test_photo, nprobe):
    """Executes the search and times the response."""
    t0 = time.perf_counter()
    meta, conf = identify_art(state, test_photo, nprobe=nprobe)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return meta, conf, latency_ms

def _score_prediction(self, true_id, meta, confidence, latency, run_id, ts):
    """Compares true ID vs predicted ID and formats the record."""
    true_id_str = str(true_id)
    pred_id = str(meta["id"]) if meta else None
    is_match = (pred_id == true_id_str)

    return {
        "run_id": run_id,
        "timestamp": ts,
        "n_features": Config.N_FEATURES,
        "nprobe": None, # Will be set in summary
        "true_id": true_id_str,
        "predicted_id": pred_id,
        "predicted_title": meta["title"] if meta else None,
        "predicted_artist": meta["artist"] if meta else None,
        "confidence": round(confidence, 4),
        "match": is_match,
        "latency_ms": latency,
    }

def _summarize_run(self, results, state, nprobe, run_id, ts):
    """Aggregates all results into a single performance row."""
    n_tested = len(results)
    if n_tested == 0:
        return {}

    correct = sum(1 for r in results if r['match'])
    avg_latency = round(sum(r["latency_ms"] for r in results) / n_tested, 2)
    
    return {
        "run_id": run_id,
        "timestamp": ts,
        "n_features": Config.N_FEATURES,
        "vault_size": len(state.source_df),
        "nprobe": nprobe,
        "n_tested": n_tested,
        "n_correct": correct,
        "accuracy": round(correct / n_tested, 4),
        "avg_latency_ms": avg_latency,
    }


def show_3panel(state, test_photo, meta, confidence, true_id):
    """Displays a 3-panel comparison with resolved path logic."""
    true_id_str = str(true_id)
    true_filename = f"{true_id_str}.jpg" if "_" in true_id_str else f"moma_{true_id_str}.jpg"
    true_blob_path = f"images/{true_filename}"
    
    true_blob = state.bucket.blob(true_blob_path)
    try:
        true_img = Image.open(BytesIO(true_blob.download_as_bytes()))
    except:
        return

    pred_img = None
    if meta:
        pred_id_str = str(meta['id'])
        pred_filename = f"{pred_id_str}.jpg" if "_" in pred_id_str else f"moma_{pred_id_str}.jpg"
        pred_blob_path = f"images/{pred_filename}"
        try:
            pred_blob = state.bucket.blob(pred_blob_path)
            pred_img = Image.open(BytesIO(pred_blob.download_as_bytes()))
        except:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(test_photo)
    axes[0].set_title("Input (Test Photo)")
    axes[0].axis('off')
    
    if pred_img:
        axes[1].imshow(pred_img)
        title = f"Prediction: {meta.get('title', 'Unknown')}\nArtist: {meta.get('artist', 'Unknown')}\nConf: {confidence:.4f}"
        axes[1].set_title(title)
    else:
        axes[1].text(0.5, 0.5, "Image Missing", ha='center', va='center')
        axes[1].set_title("Prediction (Metadata only)")
    axes[1].axis('off')
    
    axes[2].imshow(true_img)
    axes[2].set_title(f"Ground Truth\nID: {true_id_str}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
