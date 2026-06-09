import logging

from .vault.builder import load_source_metadata

logger = logging.getLogger(__name__)


def print_engine_diagnostics(state):
    logger.info("--- ARTFINDER CLOUD SYSTEM DIAGNOSTICS ---")

    try:
        df = load_source_metadata(state.bucket)
        valid_records = len(df.dropna(subset=['id']))
        logger.info("Metadata Tracking Parquet:")
        logger.info("  Total Logged Artworks: %d", valid_records)
    except Exception as e:
        logger.error("Metadata Tracking Parquet: Error loading - %s", e)

    if state.vault is not None:
        logger.info("imret Vault: loaded and ready")
    else:
        logger.info("imret Vault: not loaded")
