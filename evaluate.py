#!/usr/bin/env python3
"""
Evaluate a pre-built artfinder vault against images stored in GCS.

Downloads a random sample of stored images, applies a query transform, searches
the vault, and reports accuracy + latency. Ingestion and vault building are
handled separately by run_complete_rebuild / run_incremental_update.

Notebook usage:
    from evaluate import run_evaluation
    run_evaluation(state, n=100, transform="random", visualize=-1, display=True)

CLI usage:
    python evaluate.py --n 100 --transform wall
    python evaluate.py --transform all
"""
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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
    canvas = np.full((h + 2 * bh, w + 2 * bw), int(rng.integers(210, 246)), dtype=np.uint8)
    canvas[bh:bh + h, bw:bw + w] = warped
    return canvas


def _wall_photo(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    warped = _perspective_warp(gray, rng)
    h, w = warped.shape
    bw = int(w * float(rng.uniform(0.05, 0.15)))
    bh = int(h * float(rng.uniform(0.05, 0.15)))
    canvas = np.full((h + 2 * bh, w + 2 * bw), int(rng.integers(100, 181)), dtype=np.uint8)
    canvas[bh:bh + h, bw:bw + w] = warped
    k = int(rng.integers(0, 2)) * 2 + 1
    if k > 1:
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    return canvas


_REAL_TRANSFORMS = [_affine_jitter, _perspective_warp, _book_page, _wall_photo]

def _random_transform(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _REAL_TRANSFORMS[rng.integers(len(_REAL_TRANSFORMS))](gray, rng)

TRANSFORMS = {
    "none":        lambda g, rng: g,
    "affine":      _affine_jitter,
    "perspective": _perspective_warp,
    "book":        _book_page,
    "wall":        _wall_photo,
    "random":      _random_transform,
}


# ── Visualization ─────────────────────────────────────────────────────

def _draw_keypoint_matches(img_a, img_b, orb_config):
    orb = cv2.ORB_create(nfeatures=orb_config.max_features, scaleFactor=1.2, nlevels=8)
    kp1, des1 = orb.detectAndCompute(img_a, None)
    kp2, des2 = orb.detectAndCompute(img_b, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, 0
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good    = [m for m in matches if m.distance <= orb_config.max_hamming_distance]
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


def _save_visualization(query_gray, result_gray, original_gray, title, out_path,
                        orb_config, status, display=False):
    canvas, n_inliers = _draw_keypoint_matches(query_gray, result_gray, orb_config)
    color = "green" if status == "MATCH" else "red"

    fig = plt.figure(figsize=(26, 7))
    fig.suptitle(f"{title}  |  {n_inliers} RANSAC inliers", fontsize=9, color=color)
    gs = fig.add_gridspec(1, 4, wspace=0.05)
    ax_query  = fig.add_subplot(gs[0, 0])
    ax_ransac = fig.add_subplot(gs[0, 1:3])
    ax_orig   = fig.add_subplot(gs[0, 3])

    ax_query.imshow(query_gray, cmap="gray")
    ax_query.set_title("Transformed query", fontsize=8)
    ax_query.axis("off")

    if canvas is not None:
        ax_ransac.imshow(canvas, cmap="gray")
        ax_ransac.set_title("RANSAC keypoint match  (transformed ↔ returned)", fontsize=8)
    else:
        ax_ransac.imshow(result_gray, cmap="gray")
        ax_ransac.set_title("Returned match (no keypoints)", fontsize=8)
    ax_ransac.axis("off")

    orig_label = "Original" if status == "MATCH" else "Original (correct answer)"
    ax_orig.imshow(original_gray, cmap="gray")
    ax_orig.set_title(orig_label, fontsize=8)
    ax_orig.axis("off")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
    if display:
        try:
            from IPython.display import display as ipy_display
            ipy_display(fig)
        except ImportError:
            plt.show()
    plt.close(fig)


# ── GCS image loader ──────────────────────────────────────────────────

def _download_gray(bucket, visual_id: str) -> np.ndarray | None:
    try:
        data = bucket.blob(f"images/{visual_id}.jpg").download_as_bytes()
        arr  = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        logger.warning("Could not download %s: %s", visual_id, e)
        return None


# ── Core evaluation loop ──────────────────────────────────────────────

@dataclass
class EvalConfig:
    transform_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    orb_config:   object                # imret.OrbConfig
    visualize:    int          = 0      # number to visualize; -1 = all
    display:      bool         = False
    results_dir:  Optional[str] = None


def _run_eval_loop(vault, sample: list, sample_map: dict, eval_cfg: EvalConfig):
    rng       = np.random.default_rng(42)
    vis_limit = len(sample) if eval_cfg.visualize == -1 else eval_cfg.visualize
    vis_count = 0
    correct   = 0
    fallbacks = 0
    latencies = []

    for i, (visual_id, original_gray) in enumerate(sample):
        query  = eval_cfg.transform_fn(original_gray, rng)
        t0     = time.time()
        result = vault.search(query)
        latencies.append((time.time() - t0) * 1000)

        matched = result.label == visual_id
        if matched:              correct += 1
        if result.fallback_used: fallbacks += 1

        if vis_count < vis_limit:
            status      = "MATCH" if matched else "FAIL"
            result_gray = original_gray if matched else sample_map.get(result.label, original_gray)
            title = (
                f"[{i+1:03d}] {status}  |  "
                f"{visual_id}  →  {result.label}  conf={result.confidence:.2%}"
            )
            fname    = f"{i+1:03d}_{status.lower()}_{visual_id}.png"
            out_path = os.path.join(eval_cfg.results_dir, fname) if eval_cfg.results_dir else None
            _save_visualization(query, result_gray, original_gray,
                                title, out_path, eval_cfg.orb_config, status,
                                display=eval_cfg.display)
            vis_count += 1

    n = len(sample)
    return {
        "accuracy":  correct / n * 100,
        "fallbacks": fallbacks / n * 100,
        "avg_ms":    float(np.mean(latencies)),
        "p95_ms":    float(np.percentile(latencies, 95)),
        "correct":   correct,
        "n":         n,
    }


# ── Public API ────────────────────────────────────────────────────────

def run_evaluation(
    state,
    n: int            = 100,
    transform: str    = "random",
    visualize: int    = 0,
    display: bool     = False,
    results_dir: str  = None,
    seed: int         = 42,
):
    """
    Evaluate a pre-built vault using images already stored in GCS.

    Parameters
    ----------
    state       : SearchEngineState with bucket and (optionally) vault already loaded
    n           : number of query images to sample from GCS
    transform   : none / affine / perspective / book / wall / random / all
    visualize   : results to visualize; -1 = all
    display     : render each image inline (Colab/Jupyter)
    results_dir : directory for summary + visualizations
    seed        : random seed for sampling
    """
    import imret
    from artfinder.engine import BRAIN_PREFIX, _download_brain_from_cloud
    from artfinder.config import create_orb_config

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── Load vault if needed ──────────────────────────────────────────
    if state.vault is None:
        logger.info("Downloading vault from GCS...")
        _download_brain_from_cloud(state)
        state.vault = imret.Vault.load_from_disk(BRAIN_PREFIX, create_orb_config())
        logger.info("Vault loaded.")

    orb_config = create_orb_config()

    # ── Sample image IDs from GCS ─────────────────────────────────────
    logger.info("Listing images in GCS...")
    all_ids    = list({
        blob.name.split("/")[-1].removesuffix(".jpg")
        for blob in state.bucket.list_blobs(prefix="images/")
    })
    sample_ids = random.Random(seed).sample(all_ids, min(n, len(all_ids)))
    logger.info("Downloading %d sample images...", len(sample_ids))

    sample_map = {}
    sample     = []
    for vid in sample_ids:
        gray = _download_gray(state.bucket, vid)
        if gray is not None:
            sample_map[vid] = gray
            sample.append((vid, gray))

    logger.info("Downloaded %d images.", len(sample))

    # ── Run evaluation ────────────────────────────────────────────────
    transforms_to_run = (
        [(name, fn) for name, fn in TRANSFORMS.items() if name not in ("none", "random")]
        if transform == "all"
        else [(transform, TRANSFORMS[transform])]
    )

    sep = "=" * 56
    all_results = {}

    for name, fn in transforms_to_run:
        logger.info("Evaluating transform=%s...", name)
        sub_dir = os.path.join(results_dir, name) if results_dir else None
        if sub_dir:
            os.makedirs(sub_dir, exist_ok=True)
        eval_cfg = EvalConfig(
            transform_fn=fn,
            orb_config=orb_config,
            visualize=visualize,
            display=display,
            results_dir=sub_dir,
        )
        all_results[name] = _run_eval_loop(state.vault, sample, sample_map, eval_cfg)

    # ── Report ────────────────────────────────────────────────────────
    lines = []
    if len(all_results) == 1:
        name, r = next(iter(all_results.items()))
        s = state.vault.stats()
        lines = [
            sep,
            f"  Transform:         {name}",
            f"  Vault size:        {s['n_images']:,} images",
            f"  Evaluated:         {r['n']} samples",
            f"  Accuracy:          {r['correct']}/{r['n']}  ({r['accuracy']:.2f}%)",
            f"  Fallbacks used:    {r['fallbacks']:.1f}%",
            f"  Avg latency:       {r['avg_ms']:.2f} ms",
            f"  p95 latency:       {r['p95_ms']:.2f} ms",
            sep,
        ]
    else:
        col = 13
        header = f"  {'Transform':<12}{'Accuracy':>{col}}{'Fallback%':>{col}}{'Avg ms':>{col}}{'p95 ms':>{col}}"
        lines = [sep, header, "  " + "-" * (12 + col * 4)]
        for name, r in all_results.items():
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

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "summary.txt"), "w") as f:
            f.write(report + "\n")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from google.cloud import storage
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from artfinder.state import SearchEngineState
    from artfinder.config import Config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n",           type=int, default=100)
    parser.add_argument("--transform",   default="random", choices=list(TRANSFORMS) + ["all"])
    parser.add_argument("--visualize",   type=int, default=0, help="-1 = all")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    client = storage.Client(project=Config.PROJECT_ID)
    bucket = client.bucket(Config.BUCKET_NAME)
    state  = SearchEngineState(client=client, bucket=bucket)

    run_evaluation(
        state,
        n=args.n,
        transform=args.transform,
        visualize=args.visualize,
        results_dir=args.results_dir,
        seed=args.seed,
    )
