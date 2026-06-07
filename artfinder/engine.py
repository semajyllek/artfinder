import cv2
import numpy as np
from io import BytesIO
from datasets import load_dataset
import imret

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
    """IDs of artworks already stored in GCS, e.g. {'wikiart_12', ...}."""
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


def _ingest_item(state, item):
    """Add one artwork's vector to the vault and store its image in GCS."""
    state.vault.add(_to_grayscale(item["image"]), item["artist"])
    _upload_image(state, item["image"], item["visual_id"])


def _ingest_stream(state, stream, limit, skip_ids=frozenset()):
    """Consume the stream, ingesting new items until `limit` is hit. Returns count."""
    count = 0
    for item in stream:
        if item["visual_id"] in skip_ids:
            continue
        if count >= limit:
            break
        _ingest_item(state, item)
        count += 1
        if count % 100 == 0:
            print(f"Ingested {count} / {limit} artworks...")
    return count


def _finalize(state):
    """Re-cluster and persist locally + to cloud. Idempotent to call repeatedly."""
    print("\n🧠 Building Voronoi clusters natively in C++...")
    state.vault.build()
    state.vault.save(BRAIN_PREFIX)
    _sync_brain_to_cloud(state)


# ── Orchestrators ─────────────────────────────────────────────────────

def run_complete_rebuild(state, limit=1000, authority_set=None):
    """Wipe everything and build a fresh engine from scratch."""
    print(f"⚠️ --- STARTING COMPLETE REBUILD (Limit: {limit}) --- ⚠️\n")
    _purge_images(state)

    stream = _open_stream(authority_set)
    _ingest_stream(state, stream, limit)

    _finalize(state)
    print("\n🏆 --- COMPLETE REBUILD SUCCESSFUL --- 🏆")


def run_incremental_update(state, limit=1000, authority_set=None):
    """Append only new unique artworks, validating against what's already in GCS."""
    print(f"🚀 --- STARTING INCREMENTAL UPDATE (Limit: {limit}) --- 🚀\n")

    _download_brain_from_cloud(state)
    state.vault = imret.Vault.load_from_disk(BRAIN_PREFIX)

    known_ids = _fetch_known_ids(state)
    print(f"   • Found {len(known_ids):,} existing artworks in the database.")

    stream = _open_stream(authority_set)
    added = _ingest_stream(state, stream, limit, skip_ids=known_ids)

    if added == 0:
        print("\n⚠️ No new unique artworks found. System is fully up to date.")
        return

    _finalize(state)
    print("\n🏆 --- INCREMENTAL UPDATE COMPLETE --- 🏆")
