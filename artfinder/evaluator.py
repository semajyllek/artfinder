import cv2
import faiss
import numpy as np
import pandas as pd
from .config import Config

def load_production_brain(state):
    state.bucket.blob(Config.INDEX_PATH).download_to_filename(Config.LOCAL_INDEX)
    state.index = faiss.read_index_binary(Config.LOCAL_INDEX)
    state.bucket.blob(Config.META_PATH).download_to_filename(Config.LOCAL_META)
    state.source_df = pd.read_parquet(Config.LOCAL_META)
    state.id_map = np.zeros(state.index.ntotal, dtype='uint32')
    for idx, row in state.source_df.iterrows():
        start, end = int(row['start_row']), int(row['end_row'])
        state.id_map[start : end + 1] = idx

def tally_votes_weighted(faiss_indices, query_kp, id_map, source_df_len):
    h, w = 1000, 1000
    cx, cy = w // 2, h // 2
    weights = []
    for kp in query_kp:
        dist = np.sqrt((kp.pt[0] - cx)**2 + (kp.pt[1] - cy)**2)
        weight = np.exp(-dist**2 / (2 * (250**2)))
        weights.append(weight)
    expanded_weights = np.repeat(weights, 5)
    valid_mask = faiss_indices >= 0
    image_indices = id_map[faiss_indices[valid_mask]]
    valid_weights = expanded_weights[valid_mask]
    weighted_counts = np.bincount(image_indices, weights=valid_weights, minlength=source_df_len)
    winner_idx  = np.argmax(weighted_counts)
    confidence  = weighted_counts[winner_idx] / (np.sum(valid_weights) + 1e-6)
    return winner_idx, confidence
