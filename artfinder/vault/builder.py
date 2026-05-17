# artfinder/vault/builder.py
import os
import gc
import urllib.request
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
from ..config import Config
from ..ingestor import recover_state, vault_checkpoint, load_source_metadata

class VaultBuilder:
    """
    Modular execution engine for building the core visual database.
    Decoupled into abstract intake parsing streams and a unified feature vault loop.
    """
    def __init__(self, state):
        self.state = state

    # ─── CORE PIPELINE ENGINE ─────────────────────────────────────────────────

    def ingest_stream(self, data_stream, batch_name, total_records=None):
        """
        The central, unified ingestion gateway. Expects an iterable stream of 
        standardized dictionaries: {'visual_id', 'image', 'title', 'artist', 'filename'}
        
        Progressively builds vector data structures and automatically synchronizes
        checkpoints straight up to your active GCS bucket.
        """
        _, master_index = recover_state(self.state)
        
        try:
            current_df = load_source_metadata(self.state.bucket)
            known_ids = set(current_df['id'].dropna().unique())
        except Exception:
            known_ids = set()

        cache = []
        print(f"🚀 Initializing unified ingestion loop for batch: {batch_name}")

        for record in tqdm(data_stream, desc=f"Vaulting {batch_name}", total=total_records):
            visual_id = record['visual_id']
            if visual_id in known_ids:
                continue

            pil_img = record['image']
            if pil_img is None:
                continue

            # Process matrix values and scale descriptors to the master index
            vault_entry = self._extract_features(
                pil_img=pil_img,
                visual_id=visual_id,
                title=record['title'],
                artist=record['artist'],
                filename=record['filename'],
                master_index=master_index,
                source_label=batch_name
            )

            if vault_entry:
                cache.append(vault_entry)

            # Continuous Cloud Synchronization Checkpoint
            if len(cache) >= Config.CHECKPOINT_SIZE:
                print(f"\n💾 Flushing safe checkpoint slice to GCS. Index length: {master_index.ntotal:,}")
                vault_checkpoint(self.state, cache, master_index)
                cache = []
                gc.collect()

        # Final trailing flush
        if cache:
            vault_checkpoint(self.state, cache, master_index)

        print(f"✅ Ingestion batch complete. Master flat index scaled to: {master_index.ntotal:,} vectors.")

    def _extract_features(self, pil_img, visual_id, title, artist, filename, master_index, source_label):
        """Small focused function handling grayscale matrix conversion and ORB feature registry additions."""
        try:
            img_gray = pil_img.convert('L')
            img_gray.thumbnail(Config.RESIZE_DIM)
            
            kp, des = self.state.orb.detectAndCompute(np.array(img_gray), None)

            if des is not None:
                start_row = master_index.ntotal
                master_index.add(des)
                
                return {
                    'id': visual_id,
                    'title': title if title else 'Unlinked',
                    'artist': artist if artist else 'Unlinked',
                    'filename': filename,
                    'source': source_label,
                    'start_row': start_row,
                    'end_row': master_index.ntotal - 1
                }
        except Exception:
            return None
        return None


    # ─── FOCUSED SOURCE ADAPTER GENERATORS ────────────────────────────────────

    @staticmethod
    def parse_hf_dataset(hf_split):
        """Yields standardized dictionary footprints directly out of Hugging Face sets."""
        for idx, item in enumerate(hf_split):
            yield {
                'visual_id': f"vis_hf_{idx}",
                'image': item.get('image'),
                'title': str(item.get('title', 'Unlinked')).strip(),
                'artist': str(item.get('artist', 'Unlinked')).strip(),
                'filename': f"vis_hf_{idx}.jpg"
            }

    @staticmethod
    def parse_csv_urls(csv_path, id_col, url_col, title_col, artist_col, timeout=10):
        """Yields standard payloads by downloading runtime assets from an arbitrary CSV file."""
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            obj_id = str(row[id_col]).strip()
            url = str(row[url_col]).strip()
            
            if not url or url.lower() == 'nan':
                continue
                
            try:
                # Isolated network context wrapper
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    pil_img = Image.open(BytesIO(resp.read()))
            except Exception:
                # Cleanly bypass broken row paths or connection handshakes
                continue

            yield {
                'visual_id': f"vis_csv_{obj_id}",
                'image': pil_img,
                'title': str(row[title_col]).strip(),
                'artist': str(row[artist_col]).strip(),
                'filename': f"vis_csv_{obj_id}.jpg"
            }

    @staticmethod
    def parse_local_directory(image_dir):
        """Yields standard schemas from physical disk folder components."""
        valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
        all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
        
        for filename in all_files:
            full_path = os.path.join(image_dir, filename)
            base_name = filename.rsplit('.', 1)[0]
            
            try:
                pil_img = Image.open(full_path)
            except Exception:
                continue

            # Parse string segments using tokenized formatting rules
            artist, title = "Unknown Artist", base_name.replace('-', ' ').title()
            if "_" in base_name:
                parts = base_name.split('_', 1)
                artist = parts[0].replace('-', ' ').title()
                title = parts[1].replace('-', ' ').title()

            yield {
                'visual_id': f"vis_local_{base_name}",
                'image': pil_img,
                'title': title,
                'artist': artist,
                'filename': filename
            }


# ─── OPTIMIZED IVF COMPRESSION & GCS SYNC AT SCALE ───────────────────────────

def build_optimized_search_index(state, n_centroids=4096):
    """
    Transforms flat master index paths into an IVF Inverted cluster array.
    Trains locally to protect RAM overhead, then writes and pushes the final array to GCS.
    """
    import faiss
    
    # Bring up current master index tracking state
    _, master_index = recover_state(state)
    n_total = master_index.ntotal
    
    # IVF structural boundaries require strict density minimums for stability
    if n_total < n_centroids * 39:
        print(f"⚠️ Vector collection length ({n_total:,}) is too shallow to train {n_centroids} clusters effectively.")
        return

    print(f"Reconstructing array layouts for {n_total:,} visual features...")
    all_vectors = master_index.reconstruct_n(0, n_total)
    
    # Train Inverted File Index array structure
    quantizer = faiss.IndexBinaryFlat(Config.DIMENSION)
    index_ivf = faiss.IndexBinaryIVF(quantizer, Config.DIMENSION, n_centroids)
    
    print(f"Training IVF centers across centroid mappings...")
    index_ivf.train(all_vectors)
    index_ivf.add(all_vectors)
    
    # Save compilation binary locally
    faiss.write_index_binary(index_ivf, Config.LOCAL_INDEX)
    print("✅ Index compression completed locally.")
    
    # Sync the compiled live production binary up to your cloud bucket
    _upload_index_to_gcs(state, Config.LOCAL_INDEX, Config.INDEX_PATH)


def _upload_index_to_gcs(state, local_path, gcs_dest_path):
    """Helper module to push production optimized binaries back to cloud buckets."""
    print(f"📤 Synchronizing production IVF cluster index to GCS path: {gcs_dest_path}...")
    try:
        blob = state.bucket.blob(gcs_dest_path)
        blob.upload_from_filename(local_path)
        print("✨ GCS Synchronization complete. Production IVF brain live.")
    except Exception as e:
        print(f"❌ Critical: Cloud transfer routine failed: {e}")
