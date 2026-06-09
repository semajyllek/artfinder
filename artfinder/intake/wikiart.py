from .matcher import match_artist_signatures

def resolve_artist_string(item: dict, labels: list) -> str:
    """Extracts the integer class index from a stream row and decodes it to a string name."""
    artist_id = item.get('artist', -1)
    if 0 <= artist_id < len(labels):
        return labels[artist_id]
    return "Unknown"


def scan_authority_manifest(artist_name: str, authority_set: set, confidence_bar: int = 92) -> str:
    """Evaluates a stream name against the active authority set using fuzzy matching."""
    for curated_artist in authority_set:
        if match_artist_signatures(artist_name, curated_artist, fuzzy_threshold=confidence_bar):
            return curated_artist
    return None


def transform_to_standard_schema(idx: int, item: dict, canonical_name: str) -> dict:
    """Maps a raw Hugging Face record dictionary layout into the VaultBuilder format."""
    return {
        'visual_id': f"wikiart_{idx}",
        'title': item.get('title', 'Unknown Title'),
        'artist': canonical_name.title(), 
        'filename': f"wikiart_{idx}.jpg",
        'ImageURL': f"hf://wikiart/{idx}",
        'SourceURL': "https://www.wikiart.org",
        'Source': 'wikiart',
        'image': item['image']
    }


def wikiart_image_first_generator(stream, labels: list, authority_set: set):
    """Transforms incoming Hugging Face dataset items on-the-fly."""
    for idx, item in enumerate(stream):
        raw_artist_name = resolve_artist_string(item, labels)
        if not authority_set:
            yield transform_to_standard_schema(idx, item, raw_artist_name)
        else:
            matched_canonical_name = scan_authority_manifest(raw_artist_name, authority_set)
            if matched_canonical_name is not None:
                yield transform_to_standard_schema(idx, item, matched_canonical_name)
