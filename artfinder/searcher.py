import time
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Immutable data structure representing a production search return."""
    artwork_id: str
    title: str
    artist: str
    source_url: str
    confidence: float
    latency_ms: float
    fallback_triggered: bool = False


class ArtSearchEngine:
    """
    Combines the imret C++ vault with a fast O(1) metadata lookup
    to execute queries in milliseconds.
    """
    def __init__(self, state):
        self.state = state
        self.id_to_record = {}
        self._build_id_map()

    def _build_id_map(self):
        df = self.state.source_df
        if df is None or df.empty:
            return
        self.id_to_record = {r['id']: r for r in df.to_dict('records')}

    def find_match(self, img_np) -> SearchResult:
        start_time = time.time()

        if not isinstance(img_np, np.ndarray):
            raise ValueError("Input image must be a valid NumPy matrix array.")

        result = self.state.vault.search(img_np)
        latency_ms = (time.time() - start_time) * 1000

        unknown = SearchResult(
            artwork_id="unknown", title="Unknown", artist="Unknown",
            source_url="", confidence=0.0, latency_ms=latency_ms,
            fallback_triggered=False
        )

        if result.label == "Unknown" or result.label not in self.id_to_record:
            return unknown

        record = self.id_to_record[result.label]
        return SearchResult(
            artwork_id    = result.label,
            title         = record.get('title', 'Unknown Title'),
            artist        = record.get('artist', 'Unknown Artist'),
            source_url    = record.get('url', ''),
            confidence    = result.confidence,
            latency_ms    = latency_ms,
            fallback_triggered = result.fallback_used,
        )
