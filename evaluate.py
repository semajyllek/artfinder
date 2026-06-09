#!/usr/bin/env python3
"""
Evaluate imret on the wikiart dataset.

Ingests N images, builds the vault, saves/loads it, then evaluates accuracy
by searching with a random sample of ingested images. Mirrors the artfinder
production config (resize 800x800, 500 features).

Transforms simulate photographing the painting rather than using the original
file: affine jitter, perspective warp, book-page border, wall-photo border.

Usage:
    python evaluate.py --ingest 1000 --eval 100 --visualize -1 --results-dir results/n1000
    python evaluate.py --ingest 500 --transform all
"""
import argparse
import logging
import os
import time
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imret
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ingest",      type=int,   default=500)
parser.add_argument("--eval",        type=int,   default=100)
parser.add_argument("--batch",       type=int,   default=64)
parser.add_argument("--vault",       type=str,   default="/tmp/imret_wikiart_eval")
parser.add_argument("--results-dir", type=str,   default=None,
                    help="Directory for summary and visualizations (default: results/n<ingest>)")
parser.add_argument("--visualize",   type=int,   default=0,
                    help="Number of results to visualize; -1 = all")
parser.add_argument(
    "--transform",
    default="none",
    choices=["none", "affine", "perspective", "book", "wall", "all"],
)
args = parser.parse_args()

results_dir = args.results_dir or os.path.join("results", f"n{args.ingest}")
os.makedirs(results_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────
cfg = imret.OrbConfig()
cfg.max_features         = 500
cfg.resize_dim           = 800
cfg.fast_cells           = 8
cfg.deep_cells           = 64
cfg.max_hamming_distance = 45
cfg.confidence_threshold = 0.15


# ── Query transforms ──────────────────────────────────────────────────

def _affine_jitter(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    angle = float(rng.uniform(-15, 15))
    scale = float(rng.uniform(0.80, 1.0))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _perspective_warp(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    lim = int(min(h, w) * 0.10)
    def jitter(): return int(rng.integers(0, lim + 1))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [jitter(),         jitter()],
        [w - 1 - jitter(), jitter()],
        [w - 1 - jitter(), h - 1 - jitter()],
        [jitter(),         h - 1 - jitter()],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h))


def _book_page(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    warped = _perspective_warp(gray, rng)
    h, w = warped.shape
    bw = int(w * float(rng.uniform(0.10, 0.20)))
    bh = int(h * float(rng.uniform(0.10, 0.20)))
    page_tone = int(rng.integers(210, 246))
    canvas = np.full((h + 2 * bh, w + 2 * bw), page_tone, dtype=np.uint8)
    canvas[bh:bh + h, bw:bw + w] = warped
    return canvas


def _wall_photo(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    warped = _perspective_warp(gray, rng)
    h, w = warped.shape
    bw = int(w * float(rng.uniform(0.05, 0.15)))
    bh = int(h * float(rng.uniform(0.05, 0.15)))
    wall_tone = int(rng.integers(100, 181))
    canvas = np.full((h + 2 * bh, w + 2 * bw), wall_tone, dtype=np.uint8)
    canvas[bh:bh + h, bw:bw + w] = warped
    k = int(rng.integers(0, 2)) * 2 + 1
    if k > 1:
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    return canvas


_TRANSFORMS = {
    "none":        lambda g, rng: g,
    "affine":      _affine_jitter,
    "perspective": _perspective_warp,
    "book":        _book_page,
    "wall":        _wall_photo,
}


# ── Dataset ───────────────────────────────────────────────────────────
logger.info("Loading wikiart (streaming)...")
dataset      = load_dataset("huggan/wikiart", split="train", streaming=True)
artist_names = dataset.info.features["artist"].names

# ── Ingest ────────────────────────────────────────────────────────────
vault  = imret.Vault(cfg)
stored = {}   # visual_id → (gray_image, artist_name)

batch_images, batch_ids = [], []
t0 = time.time()
logger.info("Ingesting %d images...", args.ingest)

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
        if (idx + 1) % 200 == 0:
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
vault.save(args.vault)
t0 = time.time()
vault2 = imret.Vault.load_from_disk(args.vault, cfg)
t_load = time.time() - t0
logger.info("Load roundtrip: %.2fs", t_load)


# ── Visualization ─────────────────────────────────────────────────────

def _draw_keypoint_matches(img_a, img_b, cfg):
    orb = cv2.ORB_create(nfeatures=cfg.max_features, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, 0
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
        img_a, kp1, img_b, kp2, inliers[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return canvas, len(inliers)


def save_visualization(query_gray, result_gray, title, out_path, cfg, status):
    canvas, n_inliers = _draw_keypoint_matches(query_gray, result_gray, cfg)
    fig, axes = plt.subplots(1, 2 if canvas is None else 1, figsize=(18, 7))

    if canvas is not None:
        ax = axes if not isinstance(axes, np.ndarray) else axes[0]
        ax.imshow(canvas, cmap="gray")
        ax.set_title(f"{title}  |  {n_inliers} RANSAC inliers", fontsize=9,
                     color="green" if status == "MATCH" else "red")
        ax.axis("off")
    else:
        axes[0].imshow(query_gray, cmap="gray")
        axes[0].set_title("Query", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(result_gray, cmap="gray")
        axes[1].set_title("Returned match", fontsize=9)
        axes[1].axis("off")
        fig.suptitle(title, fontsize=9, color="green" if status == "MATCH" else "red")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate(vault, stored, transform_fn, n_eval, seed, visualize, transform_name, out_dir):
    sample_ids = random.Random(seed).sample(list(stored.keys()), n_eval)
    rng        = np.random.default_rng(seed)
    correct    = 0
    fallbacks  = 0
    latencies  = []

    vis_limit  = n_eval if visualize == -1 else visualize
    vis_count  = 0

    for i, visual_id in enumerate(sample_ids):
        gray, true_artist = stored[visual_id]
        query = transform_fn(gray, rng)

        t0 = time.time()
        result = vault.search(query)
        latencies.append((time.time() - t0) * 1000)

        matched = result.label == visual_id
        if matched:
            correct += 1
        if result.fallback_used:
            fallbacks += 1

        if vis_count < vis_limit:
            status = "MATCH" if matched else "FAIL"
            if matched:
                result_gray = gray
                result_artist = true_artist
            else:
                result_gray, result_artist = stored.get(result.label, (gray, "unknown"))

            title = (
                f"[{i+1:03d}] {status}  |  "
                f"Query: {true_artist} ({visual_id})  "
                f"→  Returned: {result_artist} ({result.label})  "
                f"conf={result.confidence:.2%}"
            )
            fname = f"{i+1:03d}_{status.lower()}_{visual_id}.png"
            save_visualization(query, result_gray, title,
                               os.path.join(out_dir, fname), cfg, status)
            vis_count += 1

    return {
        "accuracy":  correct / n_eval * 100,
        "fallbacks": fallbacks / n_eval * 100,
        "avg_ms":    float(np.mean(latencies)),
        "p95_ms":    float(np.percentile(latencies, 95)),
        "correct":   correct,
        "n":         n_eval,
    }


# ── Run ───────────────────────────────────────────────────────────────
n_eval = min(args.eval, len(stored))
seed   = 42

transforms_to_run = (
    list(_TRANSFORMS.items()) if args.transform == "all"
    else [(args.transform, _TRANSFORMS[args.transform])]
)

results = {}
for name, fn in transforms_to_run:
    logger.info("Evaluating transform=%s on %d samples...", name, n_eval)
    sub_dir = os.path.join(results_dir, name)
    os.makedirs(sub_dir, exist_ok=True)
    results[name] = evaluate(
        vault2, stored, fn, n_eval, seed,
        visualize=args.visualize,
        transform_name=name,
        out_dir=sub_dir,
    )

# ── Report ────────────────────────────────────────────────────────────
sep = "=" * 56

lines = []

if len(results) == 1:
    name, r = next(iter(results.items()))
    lines = [
        sep,
        f"  Transform:         {name}",
        f"  Vault size:        {args.ingest:,} images",
        f"  Evaluated:         {r['n']} samples",
        f"  Accuracy:          {r['correct']}/{r['n']}  ({r['accuracy']:.2f}%)",
        f"  Fallbacks used:    {r['fallbacks']:.1f}%",
        f"  Avg latency:       {r['avg_ms']:.2f} ms",
        f"  p95 latency:       {r['p95_ms']:.2f} ms",
        f"  Ingest speed:      {args.ingest / t_ingest:.1f} img/s",
        f"  Build time:        {t_build:.2f}s",
        f"  Load time:         {t_load:.2f}s",
        sep,
    ]
else:
    col = 13
    header = f"  {'Transform':<12}{'Accuracy':>{col}}{'Fallback%':>{col}}{'Avg ms':>{col}}{'p95 ms':>{col}}"
    lines = [
        sep,
        f"  Vault: {args.ingest:,} images   Eval: {n_eval} samples",
        sep,
        header,
        "  " + "-" * (12 + col * 4),
    ]
    for name, r in results.items():
        lines.append(
            f"  {name:<12}"
            f"{r['accuracy']:>{col}.1f}%"
            f"{r['fallbacks']:>{col}.1f}%"
            f"{r['avg_ms']:>{col}.2f}"
            f"{r['p95_ms']:>{col}.2f}"
        )
    lines.append(sep)

report = "\n".join(lines)
logger.info("\n%s", report)

summary_path = os.path.join(results_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(report + "\n")
logger.info("Summary written to %s", summary_path)
logger.info("Visualizations saved to %s/", results_dir)
