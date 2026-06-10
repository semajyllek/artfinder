"""
Tests for evaluate.py transforms and _run_eval_loop.

No GCS or vault required — run_evaluation() itself is integration-level
and tested manually; these cover the pure-Python logic.
"""
import os
import sys
import types
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from evaluate import (
    TRANSFORMS,
    EvalConfig,
    _run_eval_loop,
    _affine_jitter,
    _perspective_warp,
    _book_page,
    _wall_photo,
    _brightness_shift,
    _spine_warp,
    _contrast_shift,
    _random_transform,
    _REAL_TRANSFORMS,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _gray(h=200, w=200, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w), dtype=np.uint8)


def _rng(seed=42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ── EvalConfig field names ────────────────────────────────────────────

def test_eval_config_field_names():
    """Catch accidental renames of EvalConfig fields."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(EvalConfig)}
    assert fields == {"transform_fn", "orb_config", "bucket", "visualize", "display", "results_dir"}


def test_eval_config_defaults():
    fn = lambda g, rng: g
    cfg = object()
    ec = EvalConfig(transform_fn=fn, orb_config=cfg)
    assert ec.bucket is None
    assert ec.visualize == 0
    assert ec.display is False
    assert ec.results_dir is None


# ── Transforms: output shape and dtype ───────────────────────────────

@pytest.mark.parametrize("name,fn", [
    ("affine",       _affine_jitter),
    ("perspective",  _perspective_warp),
    ("book",         _book_page),
    ("wall",         _wall_photo),
    ("brightness",   _brightness_shift),
    ("spine",        _spine_warp),
    ("contrast",     _contrast_shift),
    ("random",       _random_transform),
])
def test_transform_output_is_uint8_2d(name, fn):
    img = _gray()
    out = fn(img, _rng())
    assert out.ndim == 2, f"{name}: expected 2D output, got shape {out.shape}"
    assert out.dtype == np.uint8, f"{name}: expected uint8, got {out.dtype}"


@pytest.mark.parametrize("name,fn", [
    ("affine",       _affine_jitter),
    ("perspective",  _perspective_warp),
    ("brightness",   _brightness_shift),
    ("spine",        _spine_warp),
    ("contrast",     _contrast_shift),
])
def test_transform_preserves_shape(name, fn):
    img = _gray(300, 200)
    out = fn(img, _rng())
    assert out.shape == img.shape, f"{name}: shape changed {img.shape} → {out.shape}"


@pytest.mark.parametrize("name,fn", [
    ("book",  _book_page),
    ("wall",  _wall_photo),
])
def test_border_transforms_expand_canvas(name, fn):
    img = _gray(200, 200)
    out = fn(img, _rng())
    assert out.shape[0] > img.shape[0], f"{name}: height should grow"
    assert out.shape[1] > img.shape[1], f"{name}: width should grow"


def test_none_transform_is_identity():
    img = _gray()
    out = TRANSFORMS["none"](img, _rng())
    assert np.array_equal(out, img)


def test_random_transform_uses_real_transforms():
    """random must delegate to one of the real transforms."""
    called = []
    n = len(_REAL_TRANSFORMS)
    patched = [
        lambda g, rng, i=i: (called.append(i), g)[1]
        for i in range(n)
    ]
    orig = _REAL_TRANSFORMS[:]
    try:
        _REAL_TRANSFORMS[:] = patched
        img = _gray()
        _random_transform(img, _rng())
        assert len(called) == 1
        assert called[0] in range(n)
    finally:
        _REAL_TRANSFORMS[:] = orig


def test_real_transforms_count():
    assert len(_REAL_TRANSFORMS) == 7


# ── _run_eval_loop ────────────────────────────────────────────────────

class _MockResult:
    def __init__(self, label, confidence=0.9, fallback_used=False):
        self.label        = label
        self.confidence   = confidence
        self.fallback_used = fallback_used


class _MockVault:
    def __init__(self, responses: dict):
        self._responses = responses  # visual_id → label returned

    def search(self, query):
        # key off the query pixel sum to look up the pre-registered response
        key = int(query.sum()) % 10000
        return self._responses.get(key, _MockResult("wrong"))


def _make_sample_and_vault(n=5, all_correct=True):
    rng     = np.random.default_rng(0)
    sample  = []
    sample_map = {}
    responses  = {}
    for i in range(n):
        img = rng.integers(0, 256, (100, 100), dtype=np.uint8)
        vid = f"img_{i}"
        returned_label = vid if all_correct else "wrong"
        key = int(img.sum()) % 10000
        responses[key] = _MockResult(returned_label)
        sample_map[vid] = img
        sample.append((vid, img))
    vault = _MockVault(responses)
    return vault, sample, sample_map


def _no_op_orb_config():
    cfg = types.SimpleNamespace()
    cfg.max_features         = 50
    cfg.max_hamming_distance = 45
    return cfg


def test_run_eval_loop_all_correct():
    vault, sample, sample_map = _make_sample_and_vault(n=10, all_correct=True)
    ec = EvalConfig(
        transform_fn=TRANSFORMS["none"],
        orb_config=_no_op_orb_config(),
    )
    r = _run_eval_loop(vault, sample, sample_map, ec)
    assert r["correct"] == 10
    assert r["accuracy"] == 100.0
    assert r["n"] == 10


def test_run_eval_loop_all_wrong():
    vault, sample, sample_map = _make_sample_and_vault(n=10, all_correct=False)
    ec = EvalConfig(
        transform_fn=TRANSFORMS["none"],
        orb_config=_no_op_orb_config(),
    )
    r = _run_eval_loop(vault, sample, sample_map, ec)
    assert r["correct"] == 0
    assert r["accuracy"] == 0.0


def test_run_eval_loop_result_keys():
    vault, sample, sample_map = _make_sample_and_vault(n=3)
    ec = EvalConfig(transform_fn=TRANSFORMS["none"], orb_config=_no_op_orb_config())
    r = _run_eval_loop(vault, sample, sample_map, ec)
    assert set(r.keys()) == {"accuracy", "fallbacks", "avg_ms", "p95_ms", "correct", "n"}


def test_run_eval_loop_latencies_are_positive():
    vault, sample, sample_map = _make_sample_and_vault(n=5)
    ec = EvalConfig(transform_fn=TRANSFORMS["none"], orb_config=_no_op_orb_config())
    r = _run_eval_loop(vault, sample, sample_map, ec)
    assert r["avg_ms"] > 0
    assert r["p95_ms"] >= r["avg_ms"] or np.isclose(r["p95_ms"], r["avg_ms"])


def test_run_eval_loop_fail_downloads_returned_image(monkeypatch):
    """When the vault returns a label outside the sample, the visualization
    must show the *actual* returned image (downloaded from GCS), not fall
    back to the query's own original image."""
    import evaluate

    vault, sample, sample_map = _make_sample_and_vault(n=3, all_correct=False)

    downloaded = np.full((100, 100), 123, dtype=np.uint8)
    download_calls = []

    def fake_download_gray(bucket, visual_id):
        download_calls.append(visual_id)
        return downloaded

    captured = []

    def fake_save_visualization(query_gray, result_gray, original_gray, title,
                                 out_path, orb_config, status, display=False):
        captured.append(result_gray)

    monkeypatch.setattr(evaluate, "_download_gray", fake_download_gray)
    monkeypatch.setattr(evaluate, "_save_visualization", fake_save_visualization)

    ec = evaluate.EvalConfig(
        transform_fn=TRANSFORMS["none"],
        orb_config=_no_op_orb_config(),
        bucket=object(),
        visualize=-1,
    )
    r = _run_eval_loop(vault, sample, sample_map, ec)

    assert r["correct"] == 0
    # "wrong" is not a key in sample_map, so it must be downloaded — once,
    # then served from the in-loop cache for subsequent identical labels.
    assert download_calls == ["wrong"]
    assert all(np.array_equal(rg, downloaded) for rg in captured)


def test_run_eval_loop_unknown_label_skips_download(monkeypatch):
    """An 'Unknown' result has no image to fetch — must not hit GCS."""
    import evaluate

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (100, 100), dtype=np.uint8)
    sample = [("img_0", img)]
    sample_map = {"img_0": img}

    class _UnknownVault:
        def search(self, query):
            return _MockResult("Unknown")

    download_calls = []
    monkeypatch.setattr(evaluate, "_download_gray",
                         lambda bucket, vid: download_calls.append(vid))

    ec = evaluate.EvalConfig(
        transform_fn=TRANSFORMS["none"],
        orb_config=_no_op_orb_config(),
        bucket=object(),
        visualize=-1,
    )
    _run_eval_loop(_UnknownVault(), sample, sample_map, ec)

    assert download_calls == []


def test_run_eval_loop_fallback_counting():
    n = 6
    rng = np.random.default_rng(0)
    sample = [(f"img_{i}", rng.integers(0, 256, (100, 100), dtype=np.uint8)) for i in range(n)]
    sample_map = {vid: img for vid, img in sample}

    class _FallbackVault:
        def search(self, query):
            vid = f"img_{int(query[0, 0]) % n}"
            return _MockResult(vid, fallback_used=True)

    ec = EvalConfig(transform_fn=TRANSFORMS["none"], orb_config=_no_op_orb_config())
    r = _run_eval_loop(_FallbackVault(), sample, sample_map, ec)
    assert r["fallbacks"] == 100.0
