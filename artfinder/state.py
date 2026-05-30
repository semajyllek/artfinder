import cv2
from google.cloud import storage
from dataclasses import dataclass

@dataclass
class SearchEngineState:
    """Maintains active system configurations and bucket states."""
    client: storage.Client
    bucket: storage.Bucket
    orb: cv2.Feature2D
    index: object = None
    source_df: object = None
