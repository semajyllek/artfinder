# ArtFinder: Image-First Visual Search Engine

ArtFinder is a high-performance visual search and identification engine built on top of OpenCV ORB feature descriptors and FAISS binary inverted file (IVF) clustering. The platform scales gracefully up to a 50,000-image footprint by organizing millions of local visual keypoints into distinct searchable neighborhoods.

## 🏗️ Architecture Pivot: Image-First

Historically, the system utilized a *Scraped-Data-First* architecture, which generated high network IO overhead by querying remote museum APIs (The Met, MoMA) sequentially to look up image links before performing feature extractions.

The system now runs an optimized **Image-First Pipeline**:
1. **Foundational Ingestion**: The system streams high-resolution matrices directly from local or curated image repositories (e.g., WikiArt).
2. **Visual Feature Vaulting**: Visual features are extracted instantly via an immutable CV2 ORB matrix filter and appended to a flat master binary vault (`vector_vault.bin`).
3. **Partitioned Centroid Clustering**: A K-Means calculation compiles the flat vectors into 4,096 distinct Voronoi cells to accelerate performance.
4. **Secondary Lookup Enrichment**: Museum APIs are treated as non-blocking downstream consumers. Incoming pieces of art from museum catalogs are reconciled against the existing visual database via token-based fuzzy text matching on `Title` and `Artist` strings rather than downloading duplicate images.

## 📦 Directory Structure

```text
artfinder/
├── config.py           # Global hyperparameters (500 features, 4096 clusters, 256 dimensions)
├── engine.py           # IVF training controllers and state initializations
├── ingestor.py         # Abstract base tracking and data-recovery interfaces
├── evaluator.py        # Math velocity validations and 3-way visual rendering engines
├── intake/
│   ├── wikiart.py      # Core stream parser translating category IDs to string names
│   ├── met.py          # Downstream Met metadata alignment engine
│   └── moma.py         # Downstream MoMA metadata alignment engine
└── vault/
    └── builder.py      # Unified checkpoint stream router and storage purgers
