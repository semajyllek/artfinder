# artfinder

artfinder is a visual art search system. It ingests paintings from the [WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart) on HuggingFace, stores them on Google Cloud Storage, and identifies artworks from query images using the [imret](https://github.com/semajyllek/imret) image retrieval engine.

## How it works

1. Images are streamed from the WikiArt HuggingFace dataset and filtered against an authority set of artist names.
2. Grayscale frames are ingested in batches into an `imret.Vault` via `add_batch()`.
3. After ingestion, `vault.build()` trains the FAISS IVF index.
4. The vault and a source metadata parquet (id, title, artist, url) are saved locally and uploaded to GCS.
5. At query time, the vault is loaded from GCS and searched with a query image. The matched label is resolved to a full metadata record.

## Requirements

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes `imret`. See the [imret README](https://github.com/semajyllek/imret) for build/install instructions if imret is not yet available on PyPI.

The GCS pipeline requires a Google Cloud project and service account credentials with read/write access to the storage bucket. Set the standard `GOOGLE_APPLICATION_CREDENTIALS` environment variable before running.

## Configuration

`artfinder/config.py` contains all environment constants:

```python
class Config:
    PROJECT_ID  = 'artfinder-491410'
    BUCKET_NAME = 'image-indices'

    BATCH_LIMIT     = 25000   # maximum images per ingest run
    CHECKPOINT_SIZE = 100     # progress logging interval

    META_PATH  = "system/source_metadata.parquet"   # GCS path
    LOCAL_META = "source_metadata.parquet"           # local cache
```

The imret engine parameters are defined in `create_orb_config()` in the same file:

```python
cfg.max_features         = 500
cfg.resize_dim           = 800
cfg.fast_cells           = 8
cfg.deep_cells           = 64
cfg.max_hamming_distance = 45
cfg.confidence_threshold = 0.15
```

## Local evaluation (no GCS required)

`evaluate.py` is a self-contained script that streams WikiArt directly from HuggingFace, ingests into an in-process vault, and evaluates retrieval accuracy.

```bash
python evaluate.py --ingest 2000 --eval 200 --visualize 3
```

| Flag | Default | Description |
|---|---|---|
| `--ingest N` | 500 | Number of images to ingest |
| `--eval N` | 100 | Number of images to evaluate accuracy on |
| `--batch N` | 64 | Batch size for `add_batch()` calls |
| `--vault PATH` | `/tmp/imret_wikiart_eval` | File prefix for save/load roundtrip test |
| `--visualize N` | 0 | Save RANSAC keypoint match visualizations for the first N correct results |

The script evaluates accuracy by searching with the same images that were ingested, reports accuracy, fallback rate, average and p95 latency, ingest speed, and build time. Visualizations are saved to `/tmp/imret_match_N.png`.

## Production pipeline

The production functions live in `artfinder/engine.py` and operate against GCS.

### Initial build

Purges all existing images from GCS, ingests up to `limit` images from WikiArt, builds the vault, and uploads the vault and metadata parquet to GCS.

```python
from google.cloud import storage
from artfinder.state import SearchEngineState
from artfinder.engine import run_complete_rebuild

client = storage.Client(project="artfinder-491410")
bucket = client.bucket("image-indices")
state  = SearchEngineState(client=client, bucket=bucket)

run_complete_rebuild(state, limit=10000)
```

### Incremental update

Downloads the existing vault from GCS, ingests only images not already present, rebuilds the index, and re-uploads.

```python
from artfinder.engine import run_incremental_update

run_incremental_update(state, limit=1000)
```

### Searching

```python
from artfinder.evaluator import load_production_brain
from artfinder.searcher import ArtSearchEngine
import cv2

load_production_brain(state)

engine = ArtSearchEngine(state)
gray   = cv2.imread("query.jpg", cv2.IMREAD_GRAYSCALE)
result = engine.find_match(gray)

print(result.artwork_id, result.title, result.artist, result.confidence)
```

`find_match()` returns a `SearchResult` with fields: `artwork_id`, `title`, `artist`, `source_url`, `confidence`, `latency_ms`, `fallback_triggered`.

### Diagnostics

```python
from artfinder.utils import print_engine_diagnostics

print_engine_diagnostics(state)
```

Prints a summary of the loaded state: paintings ingested, unique artists, total feature vectors, IVF cluster count, and whether the index is built. All stats are O(1) reads from in-memory structures.

### Benchmarks

```python
from artfinder.evaluator import execute_live_benchmark, run_scaling_stress_test, run_environmental_stress_test

# Accuracy and latency over a random sample from the production vault
accuracy, avg_latency = execute_live_benchmark(state, sample_size=100)

# Accuracy and latency across increasing sample sizes
run_scaling_stress_test(state, n_sizes=[10, 50, 100, 250, 500])

# Accuracy against images with simulated environmental noise
# (wall photo, book page warp, aged print colour shift, angled gallery photo)
run_environmental_stress_test(state, sample_size=50, visualize_top_n=3)
```

## Directory structure

```
artfinder/
├── config.py          # constants and imret config factory
├── engine.py          # ingestion, build, GCS transport
├── evaluator.py       # benchmarks, stress tests, visualizations
├── searcher.py        # ArtSearchEngine — search + metadata lookup
├── state.py           # SearchEngineState dataclass
├── utils.py           # print_engine_diagnostics
├── intake/
│   ├── wikiart.py     # WikiArt stream parser and schema transform
│   └── matcher.py     # fuzzy artist name matching
└── vault/
    └── builder.py     # GCS metadata parquet loader
evaluate.py            # standalone local evaluation script
```
