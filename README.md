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

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

The test suite covers the query transforms and the evaluation loop. It requires no GCS credentials — `run_evaluation` itself is integration-level and run manually against a live vault.

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

## Authority set

The authority set controls which artists are ingested. It is a list of canonical artist names stored in `data/artist_authority.json`. Only images whose artist name matches an entry in this list are ingested; when the file is absent or empty, all artists are ingested.

The default authority set (968 artists) was generated from Wikidata using sitelinks as a proxy for significance:

```bash
python generate_authority_set.py --limit 1000 --out data/artist_authority.json
```

To regenerate or expand it, adjust `--limit`. The script requires the `requests` package. The output is a sorted JSON array of strings.

You can also pass an authority set directly to either orchestrator to override the file:

```python
run_complete_rebuild(state, limit=5000, authority_set={"Rembrandt", "Vermeer", "Caravaggio"})
```

## Evaluation

`evaluate.py` evaluates a **pre-built vault** — ingestion and evaluation are separate concerns. The script downloads a random sample of images from GCS (the same images that were ingested), applies a query transform, searches the vault, and reports accuracy and latency.

Build the vault first using `run_complete_rebuild` or `run_incremental_update`, then evaluate:

```bash
python evaluate.py --n 100 --transform wall
python evaluate.py --n 200 --transform all --visualize -1 --results-dir results/
```

In a Colab notebook:

```python
# Cell 1 — build (run once)
from artfinder.engine import run_complete_rebuild
run_complete_rebuild(state, limit=1000)

# Cell 2 — evaluate (run any time)
from evaluate import run_evaluation
run_evaluation(state, n=100, transform="random", visualize=-1, display=True)
```

| Flag | Default | Description |
|---|---|---|
| `--n N` | 100 | Number of images to sample from GCS |
| `--transform` | `random` | `none` / `affine` / `perspective` / `book` / `wall` / `brightness` / `spine` / `contrast` / `random` / `all` |
| `--visualize N` | 0 | Visualizations to save/display; `-1` = all |
| `--results-dir PATH` | — | Directory for summary txt and visualizations |
| `--seed N` | 42 | Random seed for image sampling |

Transforms:

| Name | Simulates |
|---|---|
| `affine` | Rotation + scale jitter |
| `perspective` | Camera angle / corner warp |
| `book` | Perspective warp + cream page border |
| `wall` | Perspective warp + grey wall border + blur |
| `brightness` | Darker or lighter viewing conditions (gamma) |
| `spine` | Image printed across a bending book page at an angle (curve + tilt + shadow gradient) |
| `contrast` | Different camera exposure / color temperature (contrast + brightness offset) |

`random` samples uniformly across all seven real transforms, giving an unbiased accuracy estimate over all conditions.

Visualizations show a trio: transformed query | RANSAC keypoint match | original image. MATCH results are captioned in green, failures in red.

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

Pass `orb_config` to override the default imret parameters for this run:

```python
import imret

cfg = imret.OrbConfig()
cfg.max_features = 1000
cfg.resize_dim   = 1024

run_complete_rebuild(state, limit=10000, orb_config=cfg)
```

### Incremental update

Downloads the existing vault from GCS, ingests only images not already present, rebuilds the index, and re-uploads.

```python
from artfinder.engine import run_incremental_update

run_incremental_update(state, limit=1000)
```

`orb_config` is accepted here too. If omitted, the config stored in the vault's `.meta` file is used.

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

## Evaluation results

Evaluated on WikiArt images using `evaluate.py` — 100 randomly selected re-queries per vault size, `transform=none`, macOS arm64, `max_features=500`, `resize_dim=800`.

| Vault size | Accuracy | Avg latency (ms) | p95 latency (ms) | Fallback rate | Build time (s) |
|---|---|---|---|---|---|
| 100 | 100/100 (100.0%) | 10.51 | 15.35 | 0.0% | 1.4 |
| 1,000 | 100/100 (100.0%) | 10.79 | 15.20 | 0.0% | 37.3 |
| 10,000 | 100/100 (100.0%) | 14.17 | 25.65 | 0.0% | 95.3 |

Visualizations (keypoint match images for all 100 queries per vault size, labelled MATCH or FAIL) are saved to `results/` which is gitignored.

To reproduce (vault must already be built via `run_complete_rebuild`):

```bash
python evaluate.py --n 100 --visualize -1 --results-dir results/n1000
```

To evaluate all transform modes:

```bash
python evaluate.py --n 100 --transform all
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
data/
└── artist_authority.json   # curated artist name list (968 entries)
evaluate.py                 # standalone local evaluation script
generate_authority_set.py   # regenerate artist_authority.json from Wikidata
```
