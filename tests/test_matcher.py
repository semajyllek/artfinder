"""
Tests for artfinder/intake/matcher.py — name/title normalization and matching,
covering variant forms expected across WikiArt and museum collection dumps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from artfinder.intake.matcher import (
    strip_diacritics,
    strip_parentheticals,
    replace_delimiters,
    remove_punctuation,
    collapse_whitespace,
    strip_leading_article,
    strip_trailing_article,
    normalize_text_structure,
    normalize_title_structure,
    check_token_intersection,
    match_artist_signatures,
    match_artwork_titles,
)


# ── strip_diacritics ───────────────────────────────────────────────────

def test_strip_diacritics_accented_letters():
    assert strip_diacritics("Cézanne") == "Cezanne"
    assert strip_diacritics("Münch") == "Munch"
    assert strip_diacritics("Dürer") == "Durer"
    assert strip_diacritics("Gauguin") == "Gauguin"  # no change


def test_strip_diacritics_empty():
    assert strip_diacritics("") == ""
    assert strip_diacritics(None) == ""


# ── strip_parentheticals ──────────────────────────────────────────────

def test_strip_parentheticals_removes_bracketed_info():
    assert strip_parentheticals("Water Lilies (1916)").strip() == "Water Lilies"
    assert strip_parentheticals("No parens here") == "No parens here"


# ── replace_delimiters ────────────────────────────────────────────────

def test_replace_delimiters_handles_ampersand_and_separators():
    assert replace_delimiters("Mother & Child") == "Mother  and  Child"
    assert replace_delimiters("jean-baptiste_camille/corot") == "jean baptiste camille corot"


# ── remove_punctuation / collapse_whitespace ──────────────────────────

def test_remove_punctuation_keeps_alphanumeric_and_whitespace():
    assert remove_punctuation("O'Keeffe, Georgia.") == "OKeeffe Georgia"


def test_collapse_whitespace_normalizes_runs_and_trims():
    assert collapse_whitespace("  a   b\t c ") == "a b c"


# ── article stripping ─────────────────────────────────────────────────

def test_strip_leading_article():
    assert strip_leading_article("the starry night") == "starry night"
    assert strip_leading_article("a sunday afternoon") == "sunday afternoon"
    assert strip_leading_article("an actor") == "actor"
    assert strip_leading_article("starry night") == "starry night"


def test_strip_leading_article_does_not_empty_single_word_title():
    # A title that is just "The" shouldn't be reduced to nothing.
    assert strip_leading_article("the") == "the"


def test_strip_trailing_article():
    assert strip_trailing_article("starry night the") == "starry night"
    assert strip_trailing_article("starry night") == "starry night"


# ── normalize_text_structure (artist-style normalization) ─────────────

def test_normalize_text_structure_basic():
    assert normalize_text_structure("Vincent van Gogh") == "vincent van gogh"


def test_normalize_text_structure_handles_accents_punctuation_and_case():
    assert normalize_text_structure("Paul Cézanne") == "paul cezanne"
    assert normalize_text_structure("O'Keeffe, Georgia") == "okeeffe georgia"


def test_normalize_text_structure_handles_delimiters():
    assert normalize_text_structure("jean-baptiste_camille/corot") == "jean baptiste camille corot"


def test_normalize_text_structure_strips_parentheticals():
    assert normalize_text_structure("Rembrandt (workshop)") == "rembrandt"


def test_normalize_text_structure_empty_and_none():
    assert normalize_text_structure("") == ""
    assert normalize_text_structure(None) == ""


# ── normalize_title_structure ──────────────────────────────────────────

def test_normalize_title_structure_strips_leading_article():
    assert normalize_title_structure("The Starry Night") == "starry night"


def test_normalize_title_structure_strips_catalog_style_trailing_article():
    assert normalize_title_structure("Starry Night, The") == "starry night"


def test_normalize_title_structure_handles_accents_and_ampersand():
    assert normalize_title_structure("Mother & Child") == normalize_title_structure("Mother and Child")


def test_normalize_title_structure_with_no_article_unchanged():
    assert normalize_title_structure("Guernica") == "guernica"


# ── check_token_intersection ───────────────────────────────────────────

def test_check_token_intersection_handles_reordered_names():
    a = normalize_text_structure("Sloan, John French")
    b = normalize_text_structure("John French Sloan")
    assert check_token_intersection(a, b)


def test_check_token_intersection_requires_full_overlap():
    a = normalize_text_structure("Workshop of Rembrandt")
    b = normalize_text_structure("Rembrandt")
    assert not check_token_intersection(a, b)


# ── match_artist_signatures ─────────────────────────────────────────────

def test_match_artist_signatures_exact():
    assert match_artist_signatures("Vincent van Gogh", "Vincent van Gogh")


def test_match_artist_signatures_accent_variant():
    # museum dump with accents vs authority set without (or vice versa)
    assert match_artist_signatures("Paul Cezanne", "Paul Cézanne")
    assert match_artist_signatures("Paul Cézanne", "Paul Cezanne")


def test_match_artist_signatures_reordered_last_first():
    assert match_artist_signatures("Cezanne, Paul", "Paul Cezanne")


def test_match_artist_signatures_minor_typo_within_threshold():
    assert match_artist_signatures("Vincent van Gough", "Vincent van Gogh")


def test_match_artist_signatures_unrelated_names_fail():
    assert not match_artist_signatures("Pablo Picasso", "Claude Monet")


def test_match_artist_signatures_empty_inputs_fail():
    assert not match_artist_signatures("", "Vincent van Gogh")
    assert not match_artist_signatures("Vincent van Gogh", "")


# ── match_artwork_titles ─────────────────────────────────────────────────

def test_match_artwork_titles_exact():
    assert match_artwork_titles("Water Lilies", "Water Lilies")


def test_match_artwork_titles_leading_article_variant():
    assert match_artwork_titles("The Starry Night", "Starry Night")


def test_match_artwork_titles_trailing_article_catalog_form():
    assert match_artwork_titles("Starry Night, The", "The Starry Night")


def test_match_artwork_titles_accent_variant():
    assert match_artwork_titles("Dejeuner sur l'Herbe", "Déjeuner sur l'Herbe")


def test_match_artwork_titles_ampersand_vs_and():
    assert match_artwork_titles("Mother & Child", "Mother and Child")


def test_match_artwork_titles_parenthetical_date_suffix():
    assert match_artwork_titles("Water Lilies (1916)", "Water Lilies")


def test_match_artwork_titles_unrelated_titles_fail():
    assert not match_artwork_titles("Water Lilies", "The Scream")


def test_match_artwork_titles_empty_inputs_fail():
    assert not match_artwork_titles("", "Water Lilies")
    assert not match_artwork_titles("Water Lilies", "")
