import time
import requests
import pandas as pd
from tqdm.auto import tqdm
from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist

class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=1000):
        """
        Artist-First Vacuum: Iterates through the authority set and 
        harvests all available works until the limit is reached.
        """
        delta_rows = []
        headers = {'User-Agent': 'ArtFinder-ArtistFirst/1.0'}
        
        # O(1) Lookup optimization
        authority_lookup = {str(a).lower().strip() for a in self.state.authority_set}
        artists = list(self.state.authority_set)
        
        pbar = tqdm(artists, desc="Harvesting by Artist")
        
        for artist_name in pbar:
            if len(delta_rows) >= limit:
                break
                
            pbar.set_postfix({
                "harvested": f"{len(delta_rows)}/{limit}", 
                "current": artist_name[:15]
            })
            
            # 1. Get candidate IDs for this specific artist
            obj_ids = self._search_ids_by_artist(artist_name, headers)
            
            # 2. Process those IDs
            for oid in obj_ids:
                if len(delta_rows) >= limit:
                    break
                
                self._ingest_single_object(oid, known_ids, delta_rows, headers, authority_lookup)
                    
        return pd.DataFrame(delta_rows)

    def _search_ids_by_artist(self, artist_name, headers):
        """Uses the Search API to find works attributed to a specific artist."""
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/search?artistOrCulture=true&q={artist_name}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.json().get('objectIDs', []) or []
            return []
        except:
            return []

    def _ingest_single_object(self, oid, known_ids, delta_rows, headers, authority_lookup):
        """Handles GUID check and metadata retrieval for a single candidate."""
        guid = f"met_{oid}"
        if guid in known_ids:
            return

        metadata = self._get_object_metadata(oid, headers)
        if metadata and self._is_valid_asset(metadata, authority_lookup):
            self._append_to_harvest(guid, metadata, delta_rows)

    def _is_valid_asset(self, metadata, authority_lookup):
        # 1. Image Check (Fastest)
        if not metadata.get('primaryImageSmall'):
            return False

        # 2. Artist Check (The Fix)
        raw_artist = str(metadata.get('artistDisplayName', '')).lower().strip()
        if not raw_artist:
            return False
        
        # Check if any artist in our curated list appears anywhere in the Met's string
        # This turns your 0% yield into a 100% yield for valid masters.
        is_curated = any(curated_name in raw_artist for curated_name in authority_lookup)
    
        if not is_curated:
            return False

        # 3. Classification Check (Broadened)
        cls = str(metadata.get('classification', '')).lower()
        valid_types = ['paintings', 'drawings', 'watercolors', 'pastels', 'prints']
        return any(t in cls for t in valid_types)


    def _get_object_metadata(self, oid, headers):
        """Fetches raw metadata with politeness delay."""
        time.sleep(0.05) 
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            return resp.json() if resp.status_code == 200 else None
        except:
            return None

    def _append_to_harvest(self, guid, metadata, delta_rows):
        """Standardizes metadata for the vault."""
        delta_rows.append({
            'ObjectID': guid,
            'Title':    metadata.get('title', 'Unknown'),
            'Artist':   metadata.get('artistDisplayName', 'Unknown'),
            'ImageURL': metadata.get('primaryImageSmall', ''),
            'Source':   'met'
        })
