import time
import requests
import pandas as pd

from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist


class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=100):
        print("Querying The Met API for Open Access highlights...")
        
        # Identify yourself to the API
        headers = {'User-Agent': 'ArtFinder-Educational-Project/1.0 (contact: your-email@example.com)'}
        
        search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?isHighlight=true&hasImages=true&q=paintings"
        search_res = requests.get(search_url, headers=headers)
        obj_ids = search_res.json().get('objectIDs', [])
        
        delta_rows = []
        for oid in tqdm(obj_ids, desc="Scanning Met IDs"):
            if len(delta_rows) >= limit: break
            if str(oid) in known_ids: continue
            
            try:
                # Add a tiny 0.1s delay to stay under the rate limit
                time.sleep(0.1) 
                
                resp = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}", headers=headers)
                
                # Check if the response is actually JSON before decoding
                if resp.status_code != 200:
                    continue
                
                detail = resp.json()
                
                if is_curated_artist(detail.get('artistDisplayName', ''), self.state.authority_set):
                    delta_rows.append({
                        'ObjectID': oid,
                        'Title':    detail.get('title', 'Unknown'),
                        'Artist':   detail.get('artistDisplayName', 'Unknown'),
                        'ImageURL': detail.get('primaryImageSmall', ''),
                        'SourceURL': f"https://www.metmuseum.org/art/collection/search/{oid}",
                        'Source':   'met'
                    })
            except Exception as e:
                print(f"Skipping ID {oid} due to error: {e}")
                continue
        
        return pd.DataFrame(delta_rows)


