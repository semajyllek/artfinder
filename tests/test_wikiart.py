"""
Tests for artfinder/intake/wikiart.py — schema mapping and authority filtering.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from artfinder.intake.wikiart import (
    scan_authority_manifest,
    transform_to_standard_schema,
    wikiart_image_first_generator,
)


def _item(**overrides):
    base = {
        'title': 'Cornelia Street',
        'artist': 'John French Sloan',
        'date': '1920',
        'genre': 'cityscape',
        'style': 'New Realism',
        'description': 'a painting',
        'filename': '0.jpg',
        'image': 'PIL_IMAGE',
    }
    base.update(overrides)
    return base


# ── transform_to_standard_schema ──────────────────────────────────────

def test_transform_maps_core_fields():
    out = transform_to_standard_schema(0, _item(), "John French Sloan")
    assert out['visual_id'] == "wikiart_0"
    assert out['title'] == "Cornelia Street"
    assert out['artist'] == "John French Sloan"
    assert out['genre'] == "cityscape"
    assert out['style'] == "New Realism"
    assert out['date'] == "1920"
    assert out['image'] == "PIL_IMAGE"


def test_transform_uses_canonical_name_titlecased():
    out = transform_to_standard_schema(5, _item(artist="rene bertholo"), "rene bertholo")
    assert out['artist'] == "Rene Bertholo"


def test_transform_falls_back_for_missing_title():
    out = transform_to_standard_schema(1, _item(title=None), "John French Sloan")
    assert out['title'] == "Unknown Title"


def test_transform_falls_back_for_missing_genre_style_date():
    out = transform_to_standard_schema(1, _item(genre=None, style=None, date=None), "John French Sloan")
    assert out['genre'] == ""
    assert out['style'] == ""
    assert out['date'] == ""


# ── scan_authority_manifest ───────────────────────────────────────────

def test_scan_authority_manifest_matches_known_artist():
    authority = {"John French Sloan", "Rembrandt"}
    assert scan_authority_manifest("John French Sloan", authority) == "John French Sloan"


def test_scan_authority_manifest_returns_none_for_unknown_artist():
    authority = {"Rembrandt", "Vermeer"}
    assert scan_authority_manifest("John French Sloan", authority) is None


# ── wikiart_image_first_generator ─────────────────────────────────────

def test_generator_yields_all_items_with_empty_authority_set():
    stream = [_item(artist="John French Sloan"), _item(artist="Rene Bertholo")]
    out = list(wikiart_image_first_generator(stream, authority_set=set()))
    assert [o['artist'] for o in out] == ["John French Sloan", "Rene Bertholo"]
    assert [o['visual_id'] for o in out] == ["wikiart_0", "wikiart_1"]


def test_generator_filters_by_authority_set():
    stream = [_item(artist="John French Sloan"), _item(artist="Some Random Artist")]
    authority = {"John French Sloan"}
    out = list(wikiart_image_first_generator(stream, authority_set=authority))
    assert len(out) == 1
    assert out[0]['artist'] == "John French Sloan"
    # original stream index is preserved even though one item was filtered
    assert out[0]['visual_id'] == "wikiart_0"
