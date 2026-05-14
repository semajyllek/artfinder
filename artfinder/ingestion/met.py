import time
import requests
import pandas as pd

from tqdm.auto import tqdm
from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist

class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=100):
        headers = {'User-Agent': 'ArtFinder-Global/1.0'}
        delta_rows = []
        
        # We iterate through your artists
        artist_list = list(self.state.authority_set)
        
        for artist in tqdm(artist_list, desc="Deep Scanning Met"):
            if len(delta_rows) >= limit: break
            
            # Use global search to find the artist anywhere in the metadata
            search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q={artist}"
            try:
                res = requests.get(search_url, headers=headers).json()
                obj_ids = res.get('objectIDs', [])
                if not obj_ids: continue

                for oid in obj_ids[:10]: # Look deeper than top 3
                    # THE GUID FIX: check against met_ prefixed IDs
                    guid = f"met_{oid}"
                    if guid in known_ids: continue
                    
                    time.sleep(0.05)
                    d = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", headers=headers).json()
                    
                    # RELAXED FILTER: Catch anything by the artist that isn't a statue or a spoon
                    if is_curated_artist(d.get('artistDisplayName', ''), self.state.authority_set):
                        allowed_types = ['Paintings', 'Drawings', 'Watercolors', 'Pastels']
                        if any(t in str(d.get('classification')) for t in allowed_types):
                            delta_rows.append({
                                'ObjectID': guid, # Use the Prefixed ID
                                'Title':    d.get('title', 'Unknown'),
                                'Artist':   d.get('artistDisplayName', artist),
                                'ImageURL': d.get('primaryImageSmall', ''),
                                'Source':   'met'
                            })
                            if len([r for r in delta_rows if r['Artist'] == d.get('artistDisplayName')]) >= 5:
                                break
            except: continue
        return pd.DataFrame(delta_rows)
