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
    fallback_triggered: bool = False  # Tracks if this query required the wider fallback search


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
        """
        Caps features at the maximum allowed, but DOES NOT pad with fake zeros.
        This prevents empty background spaces from artificially matching.
        """
        if des is None or len(des) == 0:
            return None
            
        if len(des) > Config.N_FEATURES:
            return des[:Config.N_FEATURES]
            
        # Return exactly what it found, no fake data appended
        return des

    def _query_index_and_tally(self, standardized_des, nprobe):
        """Internal helper to execute the FAISS search and filter bad matches."""
        # Vector Subspace Pruning Check
        if hasattr(self.state.index, 'index'):
            self.state.index.index.nprobe = nprobe
        elif hasattr(self.state.index, 'nprobe'):
            self.state.index.nprobe = nprobe
            
        # Parallel C++ batch cluster query execution
        D, I = self.state.index.search(standardized_des, k=1)
        
        # Fast Inverted Identity Voting
        identity_tally = {}
        
        # SOLUTION A: Zip the distances and indices together to filter garbage matches
        max_dist = getattr(Config, 'MAX_HAMMING_DISTANCE', 45)
        
        for dist, row_idx in zip(D.flatten(), I.flatten()):
            # Filter out weak descriptors (High Hamming distance)
            if dist > max_dist:
                continue
                
            if row_idx in self.row_to_metadata_map:
                record = self.row_to_metadata_map[row_idx]
                artwork_id = record['id']
                identity_tally[artwork_id] = identity_tally.get(artwork_id, 0) + 1
                
        return identity_tally

    def find_match(self, img_np) -> SearchResult:
        """
        Executes a multi-tiered production search pass on an incoming image.
        Returns a clean SearchResult dataclass with comprehensive metadata.
        """
        start_time = time.time()
        
        if not isinstance(img_np, np.ndarray):
            raise ValueError("Input image must be a valid NumPy matrix array.")

        # 1. Feature Extraction (Performed exactly once)
        resized = cv2.resize(img_np, Config.RESIZE_DIM)
        _, des = self.state.orb.detectAndCompute(resized, None)
        standardized_des = self._standardize_query_descriptors(des)
        
        # Graceful exit if the image was entirely blank/featureless
        if standardized_des is None:
            latency_ms = (time.time() - start_time) * 1000
            return SearchResult(
                artwork_id="unknown", title="Unknown", artist="Unknown",
                source_url="", confidence=0.0, latency_ms=latency_ms,
                fallback_triggered=False
            )
            
        active_features = len(standardized_des)
        
        # 2. Tier 1 Fast Search
        fallback_triggered = False
        nprobe_primary = getattr(Config, 'NPROBE_PRIMARY', 8)
        identity_tally = self._query_index_and_tally(standardized_des, nprobe=nprobe_primary)
        
        # Calculate preliminary confidence based on *actual* feature count
        best_artwork_id = max(identity_tally, key=identity_tally.get) if identity_tally else None
        confidence = (identity_tally[best_artwork_id] / active_features) if best_artwork_id else 0.0
        
        # 3. Tier 2 Fallback Search (SOLUTION B)
        # If the top match doesn't have enough votes, re-scan with a wider net
        retry_threshold = getattr(Config, 'CONFIDENCE_RETRY_THRESHOLD', 0.15)
        
        if confidence < retry_threshold:
            fallback_triggered = True
            nprobe_fallback = getattr(Config, 'NPROBE_FALLBACK', 64)
            identity_tally = self._query_index_and_tally(standardized_des, nprobe=nprobe_fallback)
            
            # Recalculate confidence based on the wider search results
            best_artwork_id = max(identity_tally, key=identity_tally.get) if identity_tally else None
            confidence = (identity_tally[best_artwork_id] / active_features) if best_artwork_id else 0.0

        latency_ms = (time.time() - start_time) * 1000
        
        # 4. Resolve Top Matching Entity
        if not identity_tally or not best_artwork_id:
            return SearchResult(
                artwork_id="unknown", title="Unknown", artist="Unknown",
                source_url="", confidence=0.0, latency_ms=latency_ms,
                fallback_triggered=fallback_triggered
            )
            
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
            latency_ms = latency_ms,
            fallback_triggered = fallback_triggered
        )
