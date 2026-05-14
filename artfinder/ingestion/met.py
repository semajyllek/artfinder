import time
import requests
import pandas as pd

from tqdm.auto import tqdm
from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist


class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=100):
        print("Querying The Met API for all Open Access paintings...")
        
        headers = {'User-Agent': 'ArtFinder-Educational-Project/1.0'}
        # Removed isHighlight; added q=paintings to get the broadest relevant set
        search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=paintings"
        
        try:
            search_res = requests.get(search_url, headers=headers)
            obj_ids = search_res.json().get('objectIDs', [])
            print(f"Found {len(obj_ids):,} potential candidates. Filtering against authority set...")
        except Exception as e:
            print(f"Failed to connect to Met Search API: {e}")
            return pd.DataFrame()
        
        delta_rows = []
        # We process the list to find your 1,000+ expected matches
        for oid in tqdm(obj_ids, desc="Scanning Met Collection"):
            if len(delta_rows) >= limit: break
            if str(oid) in known_ids: continue
            
            try:
                # The Met API limit is ~80 requests per second; 0.05s is safe
                time.sleep(0.05) 
                resp = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", headers=headers)
                
                if resp.status_code == 200:
                    detail = resp.json()
                    # Check if the artist from the Met matches your artists.csv list
                    if is_curated_artist(detail.get('artistDisplayName', ''), self.state.authority_set):
                        delta_rows.append({
                            'ObjectID': oid,
                            'Title':    detail.get('title', 'Unknown'),
                            'Artist':   detail.get('artistDisplayName', 'Unknown'),
                            'ImageURL': detail.get('primaryImageSmall', ''),
                            'SourceURL': f"https://www.metmuseum.org/art/collection/search/{oid}",
                            'Source':   'met'
                        })
            except:
                continue
        
        return pd.DataFrame(delta_rows)

