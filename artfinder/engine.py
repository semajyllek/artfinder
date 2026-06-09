import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from datasets import load_dataset
import imret

logger = logging.getLogger(__name__)

from .config import Config, create_orb_config
from .vault.builder import load_source_metadata
from .intake.wikiart import wikiart_image_first_generator

_AUTHORITY_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "artist_authority.json")


def load_authority_set() -> set:
    path = os.path.normpath(_AUTHORITY_SET_PATH)
    if not os.path.exists(path):
        logger.warning("Authority set not found at %s — ingesting all artists.", path)
        return set()
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))

BRAIN_PREFIX = "production_brain"
BRAIN_EXTENSIONS = (".faiss", ".meta")


# ── GCS: brain transport ──────────────────────────────────────────────

def _sync_brain_to_cloud(state, prefix=BRAIN_PREFIX):
    for ext in BRAIN_EXTENSIONS:
        state.bucket.blob(f"system/{prefix}{ext}").upload_from_filename(f"{prefix}{ext}")


def _download_brain_from_cloud(state, prefix=BRAIN_PREFIX):
    for ext in BRAIN_EXTENSIONS:
        state.bucket.blob(f"system/{prefix}{ext}").download_to_filename(f"{prefix}{ext}")


# ── GCS: image asset helpers ──────────────────────────────────────────

def _purge_images(state):
    for blob in state.bucket.list_blobs(prefix="images/"):
        blob.delete()


def _fetch_known_ids(state):
    blobs = state.bucket.list_blobs(prefix="images/")
    return {blob.name.split("/")[-1].removesuffix(".jpg") for blob in blobs}


def _upload_image(state, image, visual_id):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    blob = state.bucket.blob(f"images/{visual_id}.jpg")
    blob.upload_from_string(buffer.getvalue(), content_type="image/jpeg")


# ── Ingestion primitives ──────────────────────────────────────────────

def _open_stream(authority_set=None):
    raw = load_dataset("huggan/wikiart", split="train", streaming=True)
    labels = raw.info.features["artist"].names
    authority = authority_set if authority_set is not None else load_authority_set()
    return wikiart_image_first_generator(raw, labels, authority)


def _to_grayscale(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


_BATCH_SIZE = 64


def _flush_batch(state, batch):
    images = [_to_grayscale(item["image"]) for item in batch]
    ids    = [item["visual_id"] for item in batch]
    state.vault.add_batch(images, ids)
    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(lambda item: _upload_image(state, item["image"], item["visual_id"]), batch))
    new_rows = pd.DataFrame([{
        'id':     item["visual_id"],
        'title':  item.get("title", ""),
        'artist': item.get("artist", ""),
        'url':    item.get("SourceURL", ""),
    } for item in batch])
    state.source_df = pd.concat([state.source_df, new_rows], ignore_index=True)


_PREFETCH_SIZE = 2 * _BATCH_SIZE  # items buffered ahead of the main thread
_SENTINEL = object()


def _ingest_stream(state, stream, limit, skip_ids=frozenset()):
    q = queue.Queue(maxsize=_PREFETCH_SIZE)

    def _producer():
        try:
            count = 0
            for item in stream:
                if item["visual_id"] in skip_ids:
                    continue
                if count >= limit:
                    break
                q.put(item)
                count += 1
        except Exception:
            logger.exception("Prefetch producer failed")
        finally:
            q.put(_SENTINEL)

    producer = threading.Thread(target=_producer, daemon=True)
    producer.start()

    t_start = time.time()
    t_batch = time.time()
    count = 0
    batch = []
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        batch.append(item)
        count += 1
        if len(batch) >= _BATCH_SIZE:
            _flush_batch(state, batch)
            batch = []
            if count % 100 == 0:
                elapsed = time.time() - t_start
                rate = count / elapsed if elapsed > 0 else 0
                logger.info(
                    "  [ingest] %d / %d  |  %.1f img/s  |  elapsed %.0fs",
                    count, limit, rate, elapsed,
                )
                t_batch = time.time()

    if batch:
        _flush_batch(state, batch)

    producer.join()
    elapsed = time.time() - t_start
    logger.info(
        "  [ingest] done — %d images in %.1fs  (%.1f img/s)",
        count, elapsed, count / elapsed if elapsed > 0 else 0,
    )
    return count


def _finalize(state):
    t0 = time.time()
    s = state.vault.stats()
    logger.info(
        "  [build] training IVF index — %d images, %d features, %d clusters...",
        s["n_images"], s["n_features"], s["nlist"],
    )
    state.vault.build()
    logger.info("  [build] done in %.1fs", time.time() - t0)

    t0 = time.time()
    logger.info("  [save] writing vault to disk...")
    state.vault.save(BRAIN_PREFIX)
    logger.info("  [save] done in %.1fs", time.time() - t0)

    t0 = time.time()
    logger.info("  [gcs] uploading metadata + vault...")
    state.source_df.to_parquet(Config.LOCAL_META, index=False)
    state.bucket.blob(Config.META_PATH).upload_from_filename(Config.LOCAL_META)
    _sync_brain_to_cloud(state)
    logger.info("  [gcs] done in %.1fs", time.time() - t0)


# ── Orchestrators ─────────────────────────────────────────────────────

def run_complete_rebuild(state, limit=1000, authority_set=None, orb_config=None):
    logger.info("--- STARTING COMPLETE REBUILD (Limit: %d) ---", limit)
    _purge_images(state)

    state.vault = imret.Vault(orb_config or create_orb_config())
    state.source_df = pd.DataFrame(columns=['id', 'title', 'artist', 'url'])

    stream = _open_stream(authority_set)
    _ingest_stream(state, stream, limit)

    _finalize(state)
    logger.info("--- COMPLETE REBUILD SUCCESSFUL ---")


def run_incremental_update(state, limit=1000, authority_set=None, orb_config=None):
    logger.info("--- STARTING INCREMENTAL UPDATE (Limit: %d) ---", limit)

    _download_brain_from_cloud(state)
    state.vault = imret.Vault.load_from_disk(BRAIN_PREFIX, orb_config or create_orb_config())
    state.source_df = load_source_metadata(state.bucket)

    known_ids = _fetch_known_ids(state)
    logger.info("Found %d existing artworks in the database.", len(known_ids))

    stream = _open_stream(authority_set)
    added = _ingest_stream(state, stream, limit, skip_ids=known_ids)

    if added == 0:
        logger.info("No new unique artworks found. System is fully up to date.")
        return

    _finalize(state)
    logger.info("--- INCREMENTAL UPDATE COMPLETE ---")
