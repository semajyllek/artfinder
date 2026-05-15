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

class Evaluator:
    def __init__(self, state):
        self.state = state

    def run_final_exam_and_log(self, nprobe=32, silent=True, test_ids=None):
        """Main entry point for running a validation batch."""
        if test_ids is None:
            blob = self.state.bucket.blob(Config.MANIFEST_PATH)
            test_ids = json.loads(blob.download_as_string())["test_queries"]
        
        image_results, summary_row = self.collect_eval_results(test_ids, nprobe=nprobe, silent=silent)
        
        print(f"\n--- VALIDATION SCORE: {summary_row['accuracy']*100:.1f}% | Latency: {summary_row['avg_latency_ms']}ms ---")
        return image_results, summary_row

    def collect_eval_results(self, test_ids, nprobe=32, silent=True):
        """Orchestrates the verification loop."""
        from .ingestor import load_source_metadata
        self.state.source_df = load_source_metadata(self.state.bucket)
        
        results = []
        run_id = time.strftime("%Y%m%d_%H%M%S")
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        for obj_id in tqdm(test_ids, desc="Verifying Batch"):
            # 1. Fetch
            raw_img = self._fetch_ground_truth_image(obj_id)
            if raw_img is None:
                continue

            # 2. Simulate
            test_photo = apply_simulation(raw_img)
            
            # 3. Identify
            prediction, confidence, latency = self._run_inference(test_photo, nprobe)

            # 4. Score
            result_entry = self._score_prediction(obj_id, prediction, confidence, latency, run_id, ts)
            results.append(result_entry)
            
            # 5. Visual Feedback
            if not silent or not result_entry['match']:
                show_3panel(self.state, test_photo, prediction, confidence, str(obj_id))
                    
        summary = self._summarize_run(results, nprobe, run_id, ts)
        return results, summary

    def _fetch_ground_truth_image(self, obj_id):
        """Resolves GCS path for both legacy (moma_) and new (met_) IDs."""
        oid_str = str(obj_id)
        filename = f"{oid_str}.jpg" if "_" in oid_str else f"moma_{oid_str}.jpg"
        blob_path = f"images/{filename}"
        
        try:
            blob = self.state.bucket.blob(blob_path)
            return np.array(Image.open(BytesIO(blob.download_as_bytes())).convert("RGB"))
        except Exception:
            # Silent fail for missing images to keep the loop moving
            return None

    def _run_inference(self, test_photo, nprobe):
        """Executes the search engine and times it."""
        t0 = time.perf_counter()
        meta, conf = identify_art(self.state, test_photo, nprobe=nprobe)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return meta, conf, latency_ms

    def _score_prediction(self, true_id, meta, confidence, latency, run_id, ts):
        """Compares prediction to truth and formats the dictionary."""
        true_id_str = str(true_id)
        pred_id = str(meta["id"]) if meta else None
        is_match = (pred_id == true_id_str)

        return {
            "run_id": run_id,
            "timestamp": ts,
            "n_features": Config.N_FEATURES,
            "nprobe": None, 
            "true_id": true_id_str,
            "predicted_id": pred_id,
            "predicted_title": meta["title"] if meta else None,
            "predicted_artist": meta["artist"] if meta else None,
            "confidence": round(confidence, 4),
            "match": is_match,
            "latency_ms": latency,
        }

    def _summarize_run(self, results, nprobe, run_id, ts):
        """Calculates aggregate metrics for the run."""
        n_tested = len(results)
        if n_tested == 0:
            return {"accuracy": 0, "avg_latency_ms": 0}

        correct = sum(1 for r in results if r['match'])
        avg_latency = round(sum(r["latency_ms"] for r in results) / n_tested, 2)
        
        return {
            "run_id": run_id,
            "timestamp": ts,
            "n_features": Config.N_FEATURES,
            "vault_size": len(self.state.source_df),
            "nprobe": nprobe,
            "n_tested": n_tested,
            "n_correct": correct,
            "accuracy": round(correct / n_tested, 4),
            "avg_latency_ms": avg_latency,
        }

# --- GLOBAL UTILITIES (Used by the Engine) ---

def identify_art(state, img_rgb, nprobe=32):
    """Core retrieval function using ORB and FAISS."""
    state.index.nprobe = nprobe
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kp_q, des_q = state.orb.detectAndCompute(gray, None)
    if des_q is None: return None, 0

    _, I = state.index.search(des_q, 5)
    
    # Building RAM Map logic should be in load_production_brain
    # but tally_votes_weighted is the actual decider.
    winner_idx, confidence = tally_votes_weighted(I.flatten(), kp_q, state.id_map, len(state.source_df))
    res = state.source_df.iloc[winner_idx]

    return {
        "id":     res['id'],
        "title":  res['title'],
        "artist": res['artist']
    }, confidence

def tally_votes_weighted(faiss_indices, query_kp, id_map, source_df_len):
    """Applies exponential weighting to keypoints."""
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

def apply_simulation(img_rgb):
    """Simulates real-world photo conditions."""
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

def show_3panel(state, test_photo, meta, confidence, true_id):
    """Displays visual comparison between query and prediction."""
    true_filename = f"{true_id}.jpg" if "_" in true_id else f"moma_{true_id}.jpg"
    true_blob_path = f"images/{true_filename}"
    
    try:
        blob = state.bucket.blob(true_blob_path)
        true_img = Image.open(BytesIO(blob.download_as_bytes()))
    except: return

    pred_img = None
    if meta:
        pid = str(meta['id'])
        pf = f"{pid}.jpg" if "_" in pid else f"moma_{pid}.jpg"
        try:
            p_blob = state.bucket.blob(f"images/{pf}")
            pred_img = Image.open(BytesIO(p_blob.download_as_bytes()))
        except: pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(test_photo); axes[0].set_title("Test Photo"); axes[0].axis('off')
    if pred_img:
        axes[1].imshow(pred_img)
        axes[1].set_title(f"Prediction: {meta['title']}\nConf: {confidence:.4f}")
    axes[1].axis('off')
    axes[2].imshow(true_img); axes[2].set_title(f"Ground Truth\n{true_id}"); axes[2].axis('off')
    plt.show()

def load_production_brain(state):
    """Downloads index and metadata from GCS and builds the RAM mapping."""
    print("Downloading Production Brain from GCS...")
    state.bucket.blob(Config.INDEX_PATH).download_to_filename(Config.LOCAL_INDEX)
    import faiss
    state.index = faiss.read_index_binary(Config.LOCAL_INDEX)

    state.bucket.blob(Config.META_PATH).download_to_filename(Config.LOCAL_META)
    state.source_df = pd.read_parquet(Config.LOCAL_META)

    # Build RAM Map
    state.id_map = np.zeros(state.index.ntotal, dtype='uint32')
    for idx, row in tqdm(state.source_df.iterrows(), total=len(state.source_df), desc="Mapping Vault"):
        start, end = int(row['start_row']), int(row['end_row'])
        state.id_map[start : end + 1] = idx
