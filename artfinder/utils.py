import logging

from .vault.builder import load_source_metadata

logger = logging.getLogger(__name__)


def print_engine_diagnostics(state, top_n_artists=100):
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

    if top_n_artists and state.source_df is not None and not state.source_df.empty:
        print_top_artists(state.source_df, top_n_artists)


def print_top_artists(source_df, top_n=100):
    """Prints a ranked table of the artists with the most paintings in the vault."""
    counts = source_df['artist'].value_counts().head(top_n)

    sep = "=" * 44
    print(sep)
    print(f"  TOP {len(counts)} ARTISTS BY PAINTING COUNT")
    print(sep)

    rank_w = len(str(len(counts)))
    name_w = max((len(str(name)) for name in counts.index), default=0)
    name_w = max(name_w, len("Artist"))

    print(f"  {'#':>{rank_w}}  {'Artist':<{name_w}}  Paintings")
    print(f"  {'-' * rank_w}  {'-' * name_w}  ---------")
    for i, (artist, count) in enumerate(counts.items(), start=1):
        print(f"  {i:>{rank_w}}  {artist:<{name_w}}  {count:>9,}")

    print(sep + "\n")
