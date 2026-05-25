import re
from fuzzywuzzy import fuzz

# ──────────────────────────────────────────────────────────────────────────────
# 1. TEXT CLEANING & DE-NOISING STEPPERS
# ──────────────────────────────────────────────────────────────────────────────

def strip_parentheticals(text: str) -> str:
    """Removes trailing bracketed metadata info like dates, years, or country tags."""
    if not text: return ""
    return re.sub(r'\([^)]*\)', '', text)


def replace_delimiters(text: str) -> str:
    """Converts kebab-case hyphens, underscores, or slashes to clean spaces."""
    if not text: return ""
    return text.replace('-', ' ').replace('_', ' ').replace('/', ' ')


def remove_punctuation(text: str) -> str:
    """Eliminates non-alphanumeric punctuation marks while preserving whitespace."""
    if not text: return ""
    return re.sub(r'[^\w\s]', '', text)


def normalize_text_structure(text: str) -> str:
    """
    Standardizes raw text pipelines by feeding variations sequentially 
    through cleaning cells and flattening cases.
    """
    if not text or not isinstance(text, str):
        return ""
    
    cleaned = strip_parentheticals(text)
    cleaned = replace_delimiters(cleaned)
    cleaned = remove_punctuation(cleaned)
    return cleaned.lower().strip()


# ──────────────────────────────────────────────────────────────────────────────
# 2. MATCHING STRATEGY UNITS
# ──────────────────────────────────────────────────────────────────────────────

def check_token_intersection(clean_a: str, clean_b: str) -> bool:
    """Evaluates whether two text objects share the exact same word blocks out of order."""
    tokens_a = set(clean_a.split())
    tokens_b = set(clean_b.split())
    return tokens_a == tokens_b and len(tokens_a) > 0


def calculate_token_sort_score(clean_a: str, clean_b: str) -> int:
    """Computes order-insensitive fuzzy token arrangement scores (0-100)."""
    return fuzz.token_sort_ratio(clean_a, clean_b)


def calculate_sequence_score(clean_a: str, clean_b: str) -> int:
    """Computes strict sequential Levenshtein alignment scores (0-100)."""
    ratio_score = fuzz.ratio(clean_a, clean_b)
    partial_score = fuzz.partial_ratio(clean_a, clean_b)
    return max(ratio_score, partial_score)


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOP-LEVEL UNIFIED GATEWAYS
# ──────────────────────────────────────────────────────────────────────────────

def match_artist_signatures(stream_name: str, authority_name: str, fuzzy_threshold: int = 90) -> bool:
    """
    Robust artist matching engine. Prioritizes exact alignment and 
    order-independent token checks before evaluating fuzzy permutations.
    """
    clean_stream = normalize_text_structure(stream_name)
    clean_auth = normalize_text_structure(authority_name)
    
    if not clean_stream or not clean_auth:
        return False
        
    # Check 1: Absolute Identical Matching
    if clean_stream == clean_auth:
        return True
        
    # Check 2: Order-Independent Word Block Verification (e.g., Last, First vs First Last)
    if check_token_intersection(clean_stream, clean_auth):
        return True
        
    # Check 3: Fuzzy Sorted Distance Checking
    score = calculate_token_sort_score(clean_stream, clean_auth)
    return score >= fuzzy_threshold


def match_artwork_titles(stream_title: str, target_title: str, fuzzy_threshold: int = 85) -> bool:
    """
    Robust artwork title matching engine. Focuses on string sequence 
    alignment to tolerate minor spelling typos or missing articles.
    """
    clean_stream = normalize_text_structure(stream_title)
    clean_target = normalize_text_structure(target_title)
    
    if not clean_stream or not clean_target:
        return False
        
    # Check 1: Pure Match
    if clean_stream == clean_target:
        return True
        
    # Check 2: Sequential Distance Matrix Score
    effective_score = calculate_sequence_score(clean_stream, clean_target)
    return effective_score >= fuzzy_threshold
