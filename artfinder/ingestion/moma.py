import pandas as pd

from tqdm.auto import tqdm
from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist

class MoMAIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=100):
        # Filters for items with valid images and curated artists
        potential = self.state.df_moma[self.state.df_moma['ImageURL'].str.contains(r'\.jpg|\.jpeg|\.png', case=False, na=False)].copy()
        universe = potential[potential['Artist'].apply(lambda x: is_curated_artist(x, self.state.authority_set))]
        
        delta = universe[~universe['ObjectID'].astype(str).isin(known_ids)].head(limit).copy()
        delta['Source'] = 'moma'
        delta['SourceURL'] = delta['ObjectID'].apply(lambda x: f"https://moma.org/works/{x}")
        return delta
