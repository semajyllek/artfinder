import time
import requests
import pandas as pd

from tqdm.auto import tqdm
from artfinder.ingestor import BaseIngestor
from artfinder.engine import is_curated_artist


class MetIngestor(BaseIngestor):
    def fetch_delta(self, known_ids, limit=1000):
        """
        The master loop. It forces the search through all departments 
        until it either hits the limit or runs out of IDs in the museum.
        """
        delta_rows = []
        headers = {'User-Agent': 'ArtFinder-DeepVacuum/1.0'}
        
        all_depts = self._get_all_department_ids(headers)
        
        for dept_id in all_depts:
            # Check limit at the start of every department
            if len(delta_rows) >= limit:
                break
                
            # Pass the delta_rows list into the processor to keep a running count
            self._process_department(dept_id, known_ids, limit, delta_rows, headers)
                    
        return pd.DataFrame(delta_rows)

    
    def _process_department(self, dept_id, known_ids, limit, delta_rows, headers):
        """
        LAYER 1: THE SCANNER
        Orchestrates the department walk and manages the progress bar.
        """
        obj_ids = self._get_department_ids(dept_id, headers)
        pbar = tqdm(reversed(obj_ids), desc=f"Dept {dept_id}", total=len(obj_ids))
        
        for oid in pbar:
            # Hard stop if limit is reached
            if len(delta_rows) >= limit:
                pbar.set_description(f"Dept {dept_id} [LIMIT REACHED]")
                return

            # Update postfix so you can see the harvest count live
            pbar.set_postfix({"harvested": f"{len(delta_rows)}/{limit}"})
            
            # Delegate the processing of a single ID
            self._ingest_single_object(oid, known_ids, delta_rows, headers)

    
    def _ingest_single_object(self, oid, known_ids, delta_rows, headers):
        """
        LAYER 2: THE EXTRACTOR
        Handles GUID creation, deduplication, and metadata fetching.
        """
        guid = f"met_{oid}"
        
        # Idempotency check: Skip if already in vault
        if guid in known_ids:
            return

        metadata = self._get_object_metadata(oid, headers)
        
        # Validate and transform if valid
        if metadata and self._is_valid_asset(metadata):
            self._append_to_harvest(guid, metadata, delta_rows)

    
    def _append_to_harvest(self, guid, metadata, delta_rows):
        """
        LAYER 3: THE ACCUMULATOR
        Finalizes the dictionary and adds it to the master delta list.
        """
        delta_rows.append({
            'ObjectID': guid,
            'Title':    metadata.get('title', 'Unknown'),
            'Artist':   metadata.get('artistDisplayName', 'Unknown'),
            'ImageURL': metadata.get('primaryImageSmall', ''),
            'Source':   'met'
        })

    
    def _get_all_department_ids(self, headers):
        """Fetches all 19+ departments from the Met."""
        try:
            url = "https://collectionapi.metmuseum.org/public/collection/v1/departments"
            resp = requests.get(url, headers=headers, timeout=10).json()
            return [d['departmentId'] for d in resp.get('departments', [])]
        except:
            return [11, 1, 13, 9, 21] # Fallback to major painting/print depts

    
    def _get_department_ids(self, dept_id, headers):
        """Fetches every Object ID in a specific department."""
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects?departmentIds={dept_id}"
        try:
            return requests.get(url, headers=headers, timeout=10).json().get('objectIDs', [])
        except:
            return []

    
    def _get_object_metadata(self, oid, headers):
        """Fetches metadata with a strict rate-limit delay."""
        time.sleep(0.05) 
        url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{oid}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            return resp.json() if resp.status_code == 200 else None
        except:
            return None

    
    def _is_valid_asset(self, metadata):
        """The gatekeeper: checks for curated artists and valid media."""
        artist = metadata.get('artistDisplayName', '')
        # Only process if we have a name and an image
        if not artist or not metadata.get('primaryImageSmall'):
            return False
            
        # 1. Standardize artist name and check curated list
        if not is_curated_artist(artist, self.state.authority_set):
            return False
            
        # 2. Check classification
        cls = str(metadata.get('classification', ''))
        valid_types = ['Paintings', 'Drawings', 'Watercolors', 'Pastels', 'Prints']
        return any(t in cls for t in valid_types)
