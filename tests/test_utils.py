"""
Tests for artfinder/utils.py — diagnostics output, including the
top-artists-by-painting-count table.
"""
import os
import sys
from dataclasses import dataclass

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from artfinder.utils import print_engine_diagnostics, print_top_artists


@dataclass
class _State:
    source_df: pd.DataFrame = None
    vault: object = None


def _df(rows):
    return pd.DataFrame(rows, columns=['id', 'title', 'artist', 'url', 'genre', 'style', 'date'])


def test_print_top_artists_orders_by_count_desc(capsys):
    df = _df([
        {'id': 'a', 'title': 't', 'artist': 'Rembrandt', 'url': '', 'genre': '', 'style': '', 'date': ''},
        {'id': 'b', 'title': 't', 'artist': 'Rembrandt', 'url': '', 'genre': '', 'style': '', 'date': ''},
        {'id': 'c', 'title': 't', 'artist': 'Monet', 'url': '', 'genre': '', 'style': '', 'date': ''},
    ])
    print_top_artists(df)
    out = capsys.readouterr().out

    rembrandt_pos = out.index("Rembrandt")
    monet_pos = out.index("Monet")
    assert rembrandt_pos < monet_pos
    assert "2" in out
    assert "1" in out


def test_print_top_artists_respects_top_n(capsys):
    df = _df([
        {'id': f'id_{i}', 'title': 't', 'artist': f'Artist {i}', 'url': '', 'genre': '', 'style': '', 'date': ''}
        for i in range(10)
    ])
    print_top_artists(df, top_n=3)
    out = capsys.readouterr().out
    assert "TOP 3 ARTISTS" in out


def test_print_engine_diagnostics_includes_top_artists_table(capsys):
    df = _df([
        {'id': 'a', 'title': 't', 'artist': 'Rembrandt', 'url': '', 'genre': '', 'style': '', 'date': ''},
        {'id': 'b', 'title': 't', 'artist': 'Monet', 'url': '', 'genre': '', 'style': '', 'date': ''},
    ])
    state = _State(source_df=df, vault=None)
    print_engine_diagnostics(state)
    out = capsys.readouterr().out
    assert "TOP 2 ARTISTS BY PAINTING COUNT" in out
    assert "Rembrandt" in out
    assert "Monet" in out


def test_print_engine_diagnostics_can_disable_top_artists_table(capsys):
    df = _df([
        {'id': 'a', 'title': 't', 'artist': 'Rembrandt', 'url': '', 'genre': '', 'style': '', 'date': ''},
    ])
    state = _State(source_df=df, vault=None)
    print_engine_diagnostics(state, top_n_artists=0)
    out = capsys.readouterr().out
    assert "TOP ARTISTS" not in out


def test_print_engine_diagnostics_handles_no_metadata(capsys):
    state = _State(source_df=None, vault=None)
    print_engine_diagnostics(state)
    out = capsys.readouterr().out
    assert "Metadata:            not loaded" in out
    assert "TOP" not in out
