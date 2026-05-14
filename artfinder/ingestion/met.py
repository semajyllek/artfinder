import time
import requests
import pandas as pd

from tqdm.auto import tqdm
from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist


class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=100):
        print(f"Targeting works from your {len(self.state.authority_set)} curated artists...")
        headers = {'User-Agent': 'ArtFinder-Targeted/1.0'}
        delta_rows = []
        
        # Start from a random sample of your list to ensure diversity in the vault
        artist_list = list(self.state.authority_set)
        import random
        random.shuffle(artist_list)
        
        for artist in tqdm(artist_list, desc="Searching by Artist"):
            if len(delta_rows) >= limit: break
            
            # Search specifically for that artist name
            search_url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&artistOrCulture=true&q={artist}"
            try:
                search_res = requests.get(search_url, headers=headers).json()
                obj_ids = search_res.get('objectIDs', [])
                if not obj_ids: continue
                
                # Fetch metadata for the top hits
                for oid in obj_ids[:3]: # Cap at 3 per artist for variety
                    if len(delta_rows) >= limit: break
                    if str(oid) in known_ids: continue
                    
                    time.sleep(0.05) # Rate limit safety
                    resp = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", headers=headers).json()
                    
                    # Ensure it is a painting
                    if resp.get('classification') == 'Paintings' or 'painting' in str(resp.get('objectName')).lower():
                        delta_rows.append({
                            'ObjectID': oid,
                            'Title':    resp.get('title', 'Unknown'),
                            'Artist':   resp.get('artistDisplayName', artist),
                            'ImageURL': resp.get('primaryImageSmall', ''),
                            'SourceURL': f"https://www.metmuseum.org/art/collection/search/{oid}",
                            'Source':   'met'
                        })
            except:
                continue
                
        return pd.DataFrame(delta_rows)
