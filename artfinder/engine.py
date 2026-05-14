import cv2
import pandas as pd
import re
import csv
import urllib.request
from dataclasses import dataclass
from google.cloud import storage
from google.colab import auth
from fuzzywuzzy import fuzz
from .config import Config

@dataclass
class SearchEngineState:
    client:        storage.Client
    bucket:        storage.Bucket
    df_moma:       pd.DataFrame
    authority_set: set
    orb:           cv2.Feature2D
    index:     object = None
    source_df: object = None
    id_map:    object = None

def setup_gcs():
    auth.authenticate_user()
    client = storage.Client(project=Config.PROJECT_ID)
    bucket = client.get_bucket(Config.BUCKET_NAME)
    return client, bucket

def load_moma_universe():
    MOMA_URL = "https://github.com/MuseumofModernArt/collection/raw/main/Artworks.csv?download=true"
    return pd.read_csv(MOMA_URL, low_memory=False, on_bad_lines='skip')

def build_authority_set():
    AUTH_URL = "https://raw.githubusercontent.com/oobabooga/stable-diffusion-automatic/master/artists.csv"
    with urllib.request.urlopen(AUTH_URL) as response:
        lines = [line.decode('utf-8') for line in response.readlines()]
    return {row['artist'].lower().strip() for row in csv.DictReader(lines)}

def is_curated_artist(artist_name, authority_set):
    name = str(artist_name).lower().strip()
    if not name or name == 'unknown': return False
    if name in authority_set: return True
    for auth_name in authority_set:
        if auth_name in name or name in auth_name: return True
        len_diff = abs(len(name) - len(auth_name))
        if len_diff > (1.0 - Config.FUZZY_THRESHOLD/100.0) * max(len(name), len(auth_name)):
            continue
        if fuzz.ratio(name, auth_name) > Config.FUZZY_THRESHOLD:
            return True
    return False

def initialize_engine():
    client, bucket = setup_gcs()
    df_moma   = load_moma_universe()
    auth_set  = build_authority_set()
    orb       = cv2.ORB_create(
                    nfeatures   = Config.N_FEATURES,
                    scaleFactor = Config.SCALE_FACTOR,
                    nlevels     = Config.N_LEVELS,
                    WTA_K       = Config.WTA_K,
                )
    return SearchEngineState(client, bucket, df_moma, auth_set, orb)
