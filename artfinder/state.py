from dataclasses import dataclass, field
from google.cloud import storage
import imret
import pandas as pd


@dataclass
class SearchEngineState:
    """Maintains active system configurations and bucket state."""
    client: storage.Client
    bucket: storage.Bucket
    vault: imret.Vault = None
    source_df: pd.DataFrame = None
