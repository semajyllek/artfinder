import logging

from .vault.builder import load_source_metadata

logger = logging.getLogger(__name__)


def print_engine_diagnostics(state):
    sep = "=" * 44
    print(f"\n{sep}")
    print("  ARTFINDER ENGINE DIAGNOSTICS")
    print(sep)

    if state.source_df is not None and not state.source_df.empty:
        n_images  = len(state.source_df)
        n_artists = state.source_df['artist'].nunique()
        print(f"  Paintings ingested:  {n_images:,}")
        print(f"  Unique artists:      {n_artists:,}")
    else:
        print("  Metadata:            not loaded")

    if state.vault is not None:
        try:
            s = state.vault.stats()
            print(f"  Feature vectors:     {s['n_features']:,}")
            print(f"  IVF clusters:        {s['nlist']:,}")
            print(f"  Index built:         {s['is_built']}")
        except Exception as e:
            logger.warning("Could not read vault stats: %s", e)
            print("  Vault:               loaded (stats unavailable)")
    else:
        print("  Vault:               not loaded")

    print(sep + "\n")
