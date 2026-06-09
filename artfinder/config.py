import os

class Config:
    # infrastructure
    PROJECT_ID   = 'artfinder-491410'
    BUCKET_NAME  = 'image-indices'

    # ingestion
    BATCH_LIMIT      = 25000
    CHECKPOINT_SIZE  = 100
    TIMEOUT          = 10

    # GCS paths
    META_PATH     = "system/source_metadata.parquet"
    MANIFEST_PATH = "system/test_manifest.json"

    # local cache files
    LOCAL_META  = "source_metadata.parquet"

    # eval logging
    IMAGE_RESULTS_FILE        = "artfinder_image_results.csv"
    SUMMARY_RESULTS_FILE      = "artfinder_summary_results.csv"
    ACCURACY_PLOT_FILE        = "artfinder_accuracy_plot.png"
    LATENCY_PLOT_FILE         = "artfinder_latency_plot.png"
    TOP10_ACCURACY_PLOT_FILE  = "artfinder_top10_accuracy.png"
    TOP10_LATENCY_PLOT_FILE   = "artfinder_top10_latency.png"


def create_orb_config():
    import imret
    cfg = imret.OrbConfig()
    cfg.max_features = 500
    cfg.resize_dim   = 800
    cfg.fast_cells   = 8
    cfg.deep_cells   = 64
    cfg.max_hamming_distance  = 45
    cfg.confidence_threshold  = 0.15
    return cfg
