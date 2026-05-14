import os

class Config:
    # infrastructure
    PROJECT_ID   = 'artfinder-491410'
    BUCKET_NAME  = 'image-indices'

    # feature extraction
    N_FEATURES   = 500    
    DIMENSION    = 256  # 32 * 8 bits — do not change
    RESIZE_DIM   = (800, 800)
    SCALE_FACTOR = 1.2   
    N_LEVELS     = 8     
    WTA_K        = 2

    # ingestion logic
    BATCH_LIMIT      = 25000
    CHECKPOINT_SIZE  = 100
    TIMEOUT          = 10

    # search & curation
    CLUSTERS         = 4096  
    FUZZY_THRESHOLD  = 90

    # GCS paths
    META_PATH     = "system/source_metadata.parquet"
    VAULT_PATH    = "system/vector_vault.bin"
    INDEX_PATH    = "system/search_index_ivf.bin"
    MANIFEST_PATH = "system/test_manifest.json"

    # local cache files
    LOCAL_META  = "source_metadata.parquet"
    LOCAL_VAULT = "vector_vault.bin"
    LOCAL_INDEX = "search_index.bin"

    # eval logging
    IMAGE_RESULTS_FILE   = "artfinder_image_results.csv"
    SUMMARY_RESULTS_FILE = "artfinder_summary_results.csv"
    ACCURACY_PLOT_FILE        = "artfinder_accuracy_plot.png"
    LATENCY_PLOT_FILE         = "artfinder_latency_plot.png"
    TOP10_ACCURACY_PLOT_FILE  = "artfinder_top10_accuracy.png"
    TOP10_LATENCY_PLOT_FILE   = "artfinder_top10_latency.png"

    IMAGE_FIELDNAMES = ["run_id", "timestamp", "n_features", "dimension", "resize_dim", "scale_factor", "n_levels", "wta_k", "vault_size", "clusters", "nprobe", "true_id", "predicted_id", "predicted_title", "predicted_artist", "confidence", "match", "latency_ms"]
    SUMMARY_FIELDNAMES = ["run_id", "timestamp", "n_features", "dimension", "resize_dim", "scale_factor", "n_levels", "wta_k", "vault_size", "clusters", "nprobe", "n_tested", "n_correct", "accuracy", "avg_latency_ms"]
    CONFIG_KEY_COLS = ("n_features", "dimension", "resize_dim", "scale_factor", "n_levels", "wta_k", "vault_size", "clusters", "nprobe")
