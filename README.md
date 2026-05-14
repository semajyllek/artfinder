# ArtFinder: High-Fidelity Artwork Identification

A computer vision pipeline for identifying artwork using **ORB features** and **FAISS IVF** indexing. This project utilizes pedagogical heuristics to optimize retrieval from large-scale museum datasets.

## Repository Structure
- `artfinder/config.py`: Global hyperparameters (ORB, FAISS, and GCS paths).
- `artfinder/engine.py`: Initialization, GCS authentication, and artist curation logic.
- `artfinder/ingestor.py`: Ingestion logic, feature extraction, and vault management.
- `artfinder/evaluator.py`: Retrieval evaluation, weighted voting, and RANSAC verification.

## Core Goals
1. **Curated Ingestion**: Filter sources (MoMA, AIC) using a curated artist authority list.
2. **Optimized Search**: Maintain a dataset size of 30k–50k images for sub-200ms latency.
3. **High Accuracy**: Utilize weighted voting based on keypoint distance to achieve ~98-100% accuracy.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`.
2. Run the Smoke Test notebook to verify GCS and MoMA connectivity.
3. Use `run_sync_cycle()` to add new artworks and `build_search_indices()` to train the IVF index.
