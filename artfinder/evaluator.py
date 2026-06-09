import logging
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .searcher import ArtSearchEngine
from .engine import BRAIN_PREFIX, _download_brain_from_cloud
from .vault.builder import load_source_metadata

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CORE ENGINE LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_production_brain(state):
    import imret
    from .config import create_orb_config
    logger.info("Downloading Production Brain from GCS...")
    state.source_df = load_source_metadata(state.bucket)
    _download_brain_from_cloud(state)
    state.vault = imret.Vault.load_from_disk(BRAIN_PREFIX, create_orb_config())
    logger.info("Brain loaded. Active Metadata Records: %d", len(state.source_df))


# ──────────────────────────────────────────────────────────────────────────────
# 2. BENCHMARKS
# ──────────────────────────────────────────────────────────────────────────────

def execute_live_benchmark(state, sample_size=100, verbose=True):
    """Evaluates accuracy by fetching source images from GCS and searching."""
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        if verbose:
            logger.warning("State metadata is empty. Aborting benchmark.")
        return 0.0, 0.0

    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))

    random.seed(42)
    test_samples = random.sample(valid_records, k=sample_size)
    search_engine = ArtSearchEngine(state)

    if verbose:
        logger.info("Benchmark Active: Testing %d samples...", sample_size)

    correct_matches = 0
    latencies = []

    for record in test_samples:
        img_np = _fetch_source_image(state, record['id'])
        if img_np is None:
            continue
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        start = time.time()
        result = search_engine.find_match(gray)
        latencies.append((time.time() - start) * 1000)

        if result.artwork_id == record['id']:
            correct_matches += 1

    final_accuracy = (correct_matches / len(test_samples)) * 100 if test_samples else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0

    if verbose:
        logger.info(
            "\n%s\n  ARTFINDER RUNTIME PERFORMANCE DASHBOARD\n%s\n"
            "  Total Images Evaluated:   %d\n"
            "  Total Successful Matches: %d / %d\n"
            "  Match Verification Rate:  %.2f%%\n"
            "  Average Lookup Latency:   %.2f ms\n%s",
            "=" * 52, "=" * 52,
            len(test_samples), correct_matches, len(test_samples),
            final_accuracy, avg_latency, "=" * 52,
        )

    return final_accuracy, avg_latency


def run_scaling_stress_test(state, n_sizes=[10, 50, 100, 250, 500]):
    """Runs the benchmark across scaling input sizes to verify O(1) latency."""
    logger.info("Initiating N-Size Scaling Test...")
    accuracies, latencies = [], []

    for size in n_sizes:
        logger.info("Testing Sample Size: %d...", size)
        acc, lat = execute_live_benchmark(state, sample_size=size, verbose=False)
        accuracies.append(acc)
        latencies.append(lat)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:red'
    ax1.set_xlabel('Sample Size (N)')
    ax1.set_ylabel('Average Latency (ms)', color=color)
    ax1.plot(n_sizes, latencies, marker='o', color=color, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(latencies) * 1.5)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(n_sizes, accuracies, marker='s', color=color, linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105)

    plt.title("Engine Scaling Performance (IVF Cluster Validation)")
    fig.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 3. ENVIRONMENTAL STRESS TESTS
# ──────────────────────────────────────────────────────────────────────────────

def _simulate_wall_photo(img_np):
    h, w = img_np.shape[:2]
    scale = 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    painting = cv2.resize(img_np, (new_w, new_h))

    bg_color = [random.randint(40, 220) for _ in range(3)]
    wall = np.full((h, w, 3), bg_color, dtype=np.uint8)
    noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
    wall = np.clip(wall + noise, 0, 255).astype(np.uint8)

    y_offset, x_offset = (h - new_h) // 2, (w - new_w) // 2
    wall[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = painting
    return wall


def _simulate_book_page(img_np):
    h, w = img_np.shape[:2]
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map, y_map = x_map.astype(np.float32), y_map.astype(np.float32)

    amplitude = h * 0.05
    norm_x = x_map / w
    y_map = y_map - (amplitude * np.sin(norm_x * np.pi)) + amplitude

    warped_page = cv2.remap(img_np, x_map, y_map,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(240, 245, 245))

    shadow_gradient = 0.4 + 0.6 * np.power(norm_x, 0.6)
    shadow_gradient = shadow_gradient[:, :, np.newaxis]
    warped_page = np.clip(warped_page * shadow_gradient, 0, 255).astype(np.uint8)

    bg_color = [random.randint(120, 160) for _ in range(3)]
    desk = np.full((h + int(amplitude*2), w + 40, 3), bg_color, dtype=np.uint8)
    desk[int(amplitude):int(amplitude)+h, 20:20+w] = warped_page
    return desk


def _simulate_aged_print(img_np):
    img_float = img_np.astype(np.float32)

    img_float[:, :, 0] += random.uniform(-30, 0)
    img_float[:, :, 1] += random.uniform(-10, 20)
    img_float[:, :, 2] += random.uniform(10, 40)

    img_float *= random.uniform(0.5, 1.1)

    mean = np.mean(img_float)
    img_float = (img_float - mean) * random.uniform(0.6, 0.9) + mean

    h, w = img_np.shape[:2]
    img_float += np.random.randint(-15, 15, (h, w, 3)).astype(np.float32)

    return np.clip(img_float, 0, 255).astype(np.uint8)


def _simulate_angled_gallery_photo(img_np):
    h, w = img_np.shape[:2]
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    squeeze_factor = random.uniform(0.15, 0.25)
    direction = random.choice(["left", "right", "up", "down"])
    if direction == "right":
        dst_points = np.float32([[0, 0], [w, h*squeeze_factor], [0, h], [w, h*(1-squeeze_factor)]])
    elif direction == "left":
        dst_points = np.float32([[0, h*squeeze_factor], [w, 0], [0, h*(1-squeeze_factor)], [w, h]])
    elif direction == "up":
        dst_points = np.float32([[w*squeeze_factor, 0], [w*(1-squeeze_factor), 0], [0, h], [w, h]])
    else:
        dst_points = np.float32([[0, 0], [w, 0], [w*squeeze_factor, h], [w*(1-squeeze_factor), h]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img_np, matrix, (w, h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))

    y_map, x_map = np.ogrid[:h, :w]
    dist = np.sqrt((x_map - w//2)**2 + (y_map - h//2)**2)
    gradient = np.clip(1.1 - 0.5 * (dist / np.sqrt((w//2)**2 + (h//2)**2)), 0, 1.5)[:, :, np.newaxis]
    return np.clip(warped * gradient, 0, 255).astype(np.uint8)


def _fetch_source_image(state, artwork_id):
    try:
        blob = state.bucket.blob(f"images/{artwork_id}.jpg")
        img_bytes = blob.download_as_bytes()
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", artwork_id, e)
        return None


def visualize_orb_matches(query_img, matched_img, title="", cfg=None):
    """
    Draws RANSAC-filtered ORB keypoint matches between two images.
    Uses the same ORB params as the vault so keypoints are consistent.
    """
    nfeatures  = cfg.max_features if cfg else 500
    resize_dim = cfg.resize_dim   if cfg else 800

    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8, WTA_K=2)

    def _prep(img):
        resized = cv2.resize(img, (resize_dim, resize_dim))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
        return resized, gray

    q_color, q_gray = _prep(query_img)
    m_color, m_gray = _prep(matched_img)

    kp1, des1 = orb.detectAndCompute(q_gray, None)
    kp2, des2 = orb.detectAndCompute(m_gray, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        logger.warning("Not enough keypoints to visualize.")
        return

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    max_dist = cfg.max_hamming_distance if cfg else 45
    good     = [m for m in matches if m.distance <= max_dist]

    # RANSAC: keep only geometrically consistent matches
    inliers = good
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = [m for m, keep in zip(good, mask.ravel()) if keep]

    canvas = cv2.drawMatches(
        q_color, kp1, m_color, kp2, inliers[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(18, 7))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{title}  |  {len(inliers)} RANSAC inliers  ({len(good)} Hamming-filtered)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def _apply_environmental_noise(img_np):
    scenario = random.choice(["Wall", "Book", "Aged Print"])
    if scenario == "Wall":
        return _simulate_wall_photo(img_np), scenario
    elif scenario == "Book":
        return _simulate_book_page(img_np), scenario
    else:
        return _simulate_aged_print(img_np), scenario


def run_environmental_stress_test(state, sample_size=10, visualize_top_n=3):
    """Tests the engine's resilience against geometric and environmental noise."""
    df_meta = state.source_df
    if df_meta is None or df_meta.empty:
        logger.warning("State metadata is empty. Aborting benchmark.")
        return

    valid_records = df_meta.dropna(subset=['id']).to_dict('records')
    sample_size = min(sample_size, len(valid_records))

    random.seed(int(time.time()))
    test_samples = random.sample(valid_records, k=sample_size)
    search_engine = ArtSearchEngine(state)

    correct_matches = 0
    latencies = []

    logger.info("Initiating Environmental Stress Test (%d samples)...", sample_size)

    for idx, record in enumerate(test_samples):
        artwork_id = record['id']
        original_img = _fetch_source_image(state, artwork_id)
        if original_img is None:
            continue

        mutated_img, scenario = _apply_environmental_noise(original_img)
        gray = cv2.cvtColor(mutated_img, cv2.COLOR_BGR2GRAY)

        start_time = time.time()
        result = search_engine.find_match(gray)
        latencies.append((time.time() - start_time) * 1000)

        is_correct = (result.artwork_id == artwork_id)
        if is_correct:
            correct_matches += 1

        if idx < visualize_top_n or not is_correct:
            status = "SUCCESS" if is_correct else f"FAILED (Matched: {result.artwork_id}, confidence: {result.confidence:.2f})"
            logger.info("--- Test %d: %s [%s Scenario] ---", idx + 1, status, scenario)

        if idx < visualize_top_n and result.artwork_id != "unknown":
            matched_img = _fetch_source_image(state, result.artwork_id)
            if matched_img is not None:
                cfg = state.vault._engine if hasattr(state.vault, '_engine') else None
                from .config import create_orb_config
                title = f"[{scenario}] → {result.title} by {result.artist}  conf={result.confidence:.2%}"
                visualize_orb_matches(mutated_img, matched_img, title=title,
                                      cfg=create_orb_config())

    final_accuracy = (correct_matches / len(test_samples)) * 100 if test_samples else 0.0
    avg_latency = np.mean(latencies) if latencies else 0.0

    logger.info(
        "\n%s\n  ENVIRONMENTAL STRESS TEST RESULTS\n%s\n"
        "  Images Mutated & Tested:  %d\n"
        "  Successful Matches:       %d / %d\n"
        "  Noise Survival Rate:      %.2f%%\n"
        "  Average Lookup Latency:   %.2f ms\n%s",
        "=" * 52, "=" * 52,
        len(test_samples), correct_matches, len(test_samples),
        final_accuracy, avg_latency, "=" * 52,
    )
