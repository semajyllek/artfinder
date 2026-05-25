# artfinder/intake/wikiart.py
from .matcher import match_artist_signatures

# ──────────────────────────────────────────────────────────────────────────────
# 1. TELEMETRY AND REPORTING UNIT
# ──────────────────────────────────────────────────────────────────────────────

def log_stream_heartbeat(scanned_count: int, matched_count: int):
    """Prints a periodic diagnostic heartbeat log tracking parsing velocity."""
    print(f"📊 Progress Check | Scanned: {scanned_count:,} | Successfully Matched: {matched_count:,}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. STRING LOOKUP RESOLUTION MECHANICS
# ──────────────────────────────────────────────────────────────────────────────

def resolve_artist_string(item: dict, labels: list) -> str:
    """Extracts the integer class index from a stream row and decodes it to a string name."""
    artist_id = item.get('artist', -1)
    if 0 <= artist_id < len(labels):
        return labels[artist_id]
    return "Unknown"


def scan_authority_manifest(artist_name: str, authority_set: set, confidence_bar: int = 92) -> str:
    """
    Evaluates a single stream name against the active authority set using 
    robust fuzzy signature matching. Returns the correct canonical name if found.
    """
    for curated_artist in authority_set:
        if match_artist_signatures(artist_name, curated_artist, fuzzy_threshold=confidence_bar):
            return curated_artist
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 3. SCHEMA TRANSFORMATION AND NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────

def transform_to_standard_schema(idx: int, item: dict, canonical_name: str) -> dict:
    """
    Maps a raw Hugging Face record dictionary layout into the exact 
    lowercase field key footprints expected by the VaultBuilder module.
    """
    return {
        'visual_id': f"wikiart_{idx}",
        'title': item.get('title', 'Unknown Title'),
        # Enforce clean Title Case format for the database metadata index files
        'artist': canonical_name.title(), 
        'filename': f"wikiart_{idx}.jpg",
        'ImageURL': f"hf://wikiart/{idx}",
        'SourceURL': "https://www.wikiart.org",
        'Source': 'wikiart',
        'image': item['image']
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. TOP-LEVEL STREAM GENERATOR GATEWAY
# ──────────────────────────────────────────────────────────────────────────────

def wikiart_image_first_generator(stream, labels: list, authority_set: set):
    """
    Transforms the incoming Hugging Face dataset stream by matching names 
    against a curation filter on-the-fly and emitting standardized data blocks.
    """
    print(f"\n📢 Robust Matcher Active: Evaluating stream against {len(authority_set):,} target artists...")
    
    total_scanned = 0
    total_matched = 0

    for idx, item in enumerate(stream):
        total_scanned += 1
        
        # Pull text out of the categorical record layer
        raw_artist_name = resolve_artist_string(item, labels)
        
        # Check text maps against our allowed user requirements
        matched_canonical_name = scan_authority_manifest(raw_artist_name, authority_set)
        
        # Pulse telemetry indicators every 1,000 processed stream elements
        if total_scanned % 1000 == 0:
            log_stream_heartbeat(total_scanned, total_matched)

        if matched_canonical_name is not None:
            total_matched += 1
            
            # Map parameters into our database standard structural frame
            yield transform_to_standard_schema(idx, item, matched_canonical_name)
