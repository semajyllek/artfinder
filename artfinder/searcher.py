import cv2
import time
import numpy as np
from dataclasses import dataclass
from .config import Config

@dataclass
class SearchResult:
    """Immutable data structure representing a production search return."""
    artwork_id: str
    title: str
    artist: str
    source_url: str
    confidence: float
    latency_ms: float


class ArtSearchEngine:
    """
    Dedicated production search coordinator. Combines C++ vector calculations 
    with a fast O(1) index map to execute queries in milliseconds.
    """
    def __init__(self, state):
        self.state = state
        self.row_to_metadata_map = {}
        self._build_inverted_index_map()

    def _build_inverted_index_map(self):
        """Compiles row boundaries into a direct hash map for O(1) lookups."""
        df_meta = self.state.source_df
        if df_meta is None or df_meta.empty:
            return

        # Convert dataframe array rows into a high-speed Python dictionary
        valid_records = df_meta.dropna(subset=['id']).to_dict('records')
        for record in valid_records:
            s_row = int(record['start_row'])
            e_row = int(record['end_row'])
            
            # Map every single vector index to its parent record
            for row_idx in range(s_row, e_row + 1):
                self.row_to_metadata_map[row_idx] = record

    def _standardize_query_descriptors(self, des):
        """Ensures query shapes strictly match FAISS matrix constraints."""
        if des is None or len(des) == 0:
            return np.zeros((Config.N_FEATURES, 32), dtype=np.uint8)
        if len(des) > Config.N_FEATURES:
            return des[:Config.N_FEATURES]
        elif len(des) < Config.N_FEATURES:
            padding = np.zeros((Config.N_FEATURES - len(des), 32), dtype=np.uint8)
            return np.vstack([des, padding])
        return des

    def find_match(self, img_np, nprobe=8) -> SearchResult:
        """
        Executes an ultra-fast production search pass on an incoming image.
        Returns a clean SearchResult dataclass with comprehensive metadata.
        """
        start_time = time.time()
        
        if not isinstance(img_np, np.ndarray):
            raise ValueError("Input image must be a valid NumPy matrix array.")

        # 1. Feature Extraction
        resized = cv2.resize(img_np, Config.RESIZE_DIM)
        _, des = self.state.orb.detectAndCompute(resized, None)
        standardized_des = self._standardize_query_descriptors(des)
        
        # 2. Vector Subspace Pruning Check
        if hasattr(self.state.index, 'nprobe'):
            self.state.index.nprobe = nprobe
            
        # Parallel C++ batch cluster query execution
        D, I = self.state.index.search(standardized_des, k=1)
        latency_ms = (time.time() - start_time) * 1000
        
        # 3. Fast Inverted Identity Voting
        identity_tally = {}
        for row_idx in I.flatten():
            if row_idx in self.row_to_metadata_map:
                record = self.row_to_metadata_map[row_idx]
                artwork_id = record['id']
                identity_tally[artwork_id] = identity_tally.get(artwork_id, 0) + 1
                
        # 4. Resolve Top Matching Entity
        if not identity_tally:
            return SearchResult(
                artwork_id="unknown", title="Unknown", artist="Unknown",
                source_url="", confidence=0.0, latency_ms=latency_ms
            )
            
        best_artwork_id = max(identity_tally, key=identity_tally.get)
        confidence = identity_tally[best_artwork_id] / Config.N_FEATURES
        
        # Pull clean canonical descriptive attributes from the mapped record
        matched_record = next(
            r for r in self.row_to_metadata_map.values() if r['id'] == best_artwork_id
        )
        
        return SearchResult(
            artwork_id = best_artwork_id,
            title      = matched_record.get('title', 'Unknown Title'),
            artist     = matched_record.get('artist', 'Unknown Artist'),
            source_url = matched_record.get('url', ''),
            confidence = confidence,
            latency_ms = latency_ms
        )
