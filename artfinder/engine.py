import logging
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

def _open_stream(authority_set):
    raw = load_dataset("huggan/wikiart", split="train", streaming=True)
    labels = raw.info.features["artist"].names
    authority = authority_set if authority_set is not None else set()
    return wikiart_image_first_generator(raw, labels, authority)


def _to_grayscale(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


_BATCH_SIZE = 64


def _flush_batch(state, batch):
    images = [_to_grayscale(item["image"]) for item in batch]
    ids    = [item["visual_id"] for item in batch]
    state.vault.add_batch(images, ids)
    for item in batch:
        _upload_image(state, item["image"], item["visual_id"])
    new_rows = pd.DataFrame([{
        'id':     item["visual_id"],
        'title':  item.get("title", ""),
        'artist': item.get("artist", ""),
        'url':    item.get("SourceURL", ""),
    } for item in batch])
    state.source_df = pd.concat([state.source_df, new_rows], ignore_index=True)


def _ingest_stream(state, stream, limit, skip_ids=frozenset()):
    count = 0
    batch = []
    for item in stream:
        if item["visual_id"] in skip_ids:
            continue
        if count >= limit:
            break
        batch.append(item)
        count += 1
        if len(batch) >= _BATCH_SIZE:
            _flush_batch(state, batch)
            batch = []
            if count % 100 == 0:
                logger.info("Ingested %d / %d artworks...", count, limit)
    if batch:
        _flush_batch(state, batch)
    return count


def _finalize(state):
    logger.info("Building Voronoi clusters natively in C++...")
    state.vault.build()
    state.vault.save(BRAIN_PREFIX)
    state.source_df.to_parquet(Config.LOCAL_META, index=False)
    state.bucket.blob(Config.META_PATH).upload_from_filename(Config.LOCAL_META)
    _sync_brain_to_cloud(state)


# ── Orchestrators ─────────────────────────────────────────────────────

def run_complete_rebuild(state, limit=1000, authority_set=None):
    logger.info("--- STARTING COMPLETE REBUILD (Limit: %d) ---", limit)
    _purge_images(state)

    state.vault = imret.Vault(create_orb_config())
    state.source_df = pd.DataFrame(columns=['id', 'title', 'artist', 'url'])

    stream = _open_stream(authority_set)
    _ingest_stream(state, stream, limit)

    _finalize(state)
    logger.info("--- COMPLETE REBUILD SUCCESSFUL ---")


def run_incremental_update(state, limit=1000, authority_set=None):
    logger.info("--- STARTING INCREMENTAL UPDATE (Limit: %d) ---", limit)

    _download_brain_from_cloud(state)
    state.vault = imret.Vault.load_from_disk(BRAIN_PREFIX, create_orb_config())
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
