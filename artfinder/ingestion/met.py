import requests
import pandas as pd
from .ingestor import BaseIngestor
from .engine import is_curated_artist

class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=100):
        print("Querying The Met API for Open Access highlights...")
        # Search for artworks with images that are part of the 'highlights'
        search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?isHighlight=true&hasImages=true&q=paintings"
        obj_ids = requests.get(search_url).json().get('objectIDs', [])
        
        delta_rows = []
        for oid in obj_ids:
            if len(delta_rows) >= limit: break
            if str(oid) in known_ids: continue
            
            # Fetch specific metadata
            detail = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}").json()
            
            # Use your existing curated artist logic
            if is_curated_artist(detail.get('artistDisplayName', ''), self.state.authority_set):
                delta_rows.append({
                    'ObjectID': oid,
                    'Title':    detail.get('title', 'Unknown'),
                    'Artist':   detail.get('artistDisplayName', 'Unknown'),
                    'ImageURL': detail.get('primaryImageSmall', ''),
                    'SourceURL': f"https://www.metmuseum.org/art/collection/search/{oid}",
                    'Source':   'met'
                })
        
        return pd.DataFrame(delta_rows)
