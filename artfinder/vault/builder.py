import pandas as pd
from io import BytesIO
from ..config import Config


def load_source_metadata(bucket):
    blob = bucket.blob(Config.META_PATH)
    if blob.exists():
        content = blob.download_as_bytes()
        return pd.read_parquet(BytesIO(content))
    return pd.DataFrame(columns=['id', 'title', 'artist', 'url'])
