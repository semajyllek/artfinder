#!/usr/bin/env python3
"""
Evaluate imret on the wikiart dataset.

Ingests N images, builds the vault, saves/loads it, then evaluates
accuracy by searching with the same images. Mirrors the artfinder
production config (resize 800x800, 500 features, etc.).

Usage:
    python evaluate.py [--ingest N] [--eval N] [--batch N]
"""
import argparse
import logging
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imret
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ingest", type=int, default=500,  help="Images to ingest")
parser.add_argument("--eval",   type=int, default=100,  help="Images to evaluate")
parser.add_argument("--batch",  type=int, default=64,   help="add_batch() chunk size")
parser.add_argument("--vault",     type=str, default="/tmp/imret_wikiart_eval", help="Vault file prefix")
parser.add_argument("--visualize", type=int, default=0, help="Show keypoint matches for first N correct results")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────
cfg = imret.OrbConfig()
cfg.max_features         = 500
cfg.resize_dim           = 800
cfg.fast_cells           = 8
cfg.deep_cells           = 64
cfg.max_hamming_distance = 45
cfg.confidence_threshold = 0.15

# ── Load dataset ──────────────────────────────────────────────────────
logger.info("Loading wikiart (streaming)...")
dataset      = load_dataset("huggan/wikiart", split="train", streaming=True)
artist_names = dataset.info.features["artist"].names

# ── Ingest ────────────────────────────────────────────────────────────
vault  = imret.Vault(cfg)
stored = {}   # visual_id → (gray_image, artist_name)

batch_images, batch_ids = [], []
t0 = time.time()

logger.info("Ingesting %d images in batches of %d...", args.ingest, args.batch)
for idx, item in enumerate(dataset):
    if idx >= args.ingest:
        break

    visual_id   = f"wikiart_{idx}"
    artist_id   = item.get("artist", -1)
    artist_name = artist_names[artist_id] if 0 <= artist_id < len(artist_names) else "Unknown"

    rgb  = np.array(item["image"].convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    batch_images.append(gray)
    batch_ids.append(visual_id)
    stored[visual_id] = (gray, artist_name)

    if len(batch_images) >= args.batch:
        vault.add_batch(batch_images, batch_ids)
        batch_images, batch_ids = [], []
        if (idx + 1) % 100 == 0:
            logger.info("  %d / %d", idx + 1, args.ingest)

if batch_images:
    vault.add_batch(batch_images, batch_ids)

t_ingest = time.time() - t0
logger.info("  Done. %.1fs  (%.1f img/s)", t_ingest, args.ingest / t_ingest)

# ── Build ─────────────────────────────────────────────────────────────
logger.info("Building index...")
t0 = time.time()
vault.build()
t_build = time.time() - t0
logger.info("  Done. %.2fs", t_build)

# ── Save + Load roundtrip ─────────────────────────────────────────────
logger.info("Saving to %s...", args.vault)
vault.save(args.vault)
logger.info("Loading back from disk...")
t0 = time.time()
vault2 = imret.Vault.load_from_disk(args.vault, cfg)
logger.info("  Load time: %.2fs", time.time() - t0)

# ── Visualization ────────────────────────────────────────────────────

def visualize_match(query_gray, matched_gray, title, cfg):
    orb = cv2.ORB_create(nfeatures=cfg.max_features, scaleFactor=1.2, nlevels=8, WTA_K=2)

    kp1, des1 = orb.detectAndCompute(query_gray, None)
    kp2, des2 = orb.detectAndCompute(matched_gray, None)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good    = [m for m in matches if m.distance <= cfg.max_hamming_distance]

    inliers = good
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = [m for m, keep in zip(good, mask.ravel()) if keep]

    canvas = cv2.drawMatches(
        query_gray, kp1, matched_gray, kp2, inliers[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    out_path = f"/tmp/imret_match_{shown + 1}.png"
    plt.figure(figsize=(18, 7))
    plt.imshow(canvas, cmap="gray")
    plt.title(f"{title}  |  {len(inliers)} RANSAC inliers")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.show()
    logger.info("  Saved visualization to %s", out_path)


# ── Evaluate ──────────────────────────────────────────────────────────
n_eval      = min(args.eval, len(stored))
sample_ids  = random.sample(list(stored.keys()), n_eval)

logger.info("Evaluating on %d samples...", n_eval)
correct    = 0
fallbacks  = 0
latencies  = []
shown      = 0

for visual_id in sample_ids:
    gray, artist_name = stored[visual_id]
    t0 = time.time()
    result = vault2.search(gray)
    latencies.append((time.time() - t0) * 1000)

    matched = result.label == visual_id
    if matched:
        correct += 1
    if result.fallback_used:
        fallbacks += 1

    if matched and shown < args.visualize and result.label in stored:
        matched_gray, matched_artist = stored[result.label]
        title = f"Query: {artist_name} ({visual_id})  ->  Match: {matched_artist}  conf={result.confidence:.2%}"
        visualize_match(gray, matched_gray, title, cfg)
        shown += 1

accuracy  = correct / n_eval * 100
avg_lat   = np.mean(latencies)
p95_lat   = np.percentile(latencies, 95)

logger.info(
    "\n%s\n"
    "  Vault size:        %d images\n"
    "  Evaluated:         %d samples\n"
    "  Accuracy:          %d/%d  (%.2f%%)\n"
    "  Fallbacks used:    %d/%d  (%.1f%%)\n"
    "  Avg latency:       %.2f ms\n"
    "  p95 latency:       %.2f ms\n"
    "  Ingest speed:      %.1f img/s\n"
    "  Build time:        %.2fs\n%s",
    "=" * 52,
    args.ingest, n_eval,
    correct, n_eval, accuracy,
    fallbacks, n_eval, fallbacks / n_eval * 100,
    avg_lat, p95_lat,
    args.ingest / t_ingest, t_build,
    "=" * 52,
)
