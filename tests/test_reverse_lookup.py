"""Tests for reverse_lookup."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from demo_utils import load_model, reset_lookup_tables, reverse_lookup


def setup_function() -> None:
    reset_lookup_tables()


def test_reverse_lookup_verified_gold_false_friend() -> None:
    hits = reverse_lookup("ತರಿ", "kn", model=load_model())
    assert hits
    match = next(h for h in hits if h["other_word"] == "తరి")
    assert match["source"] == "verified"
    assert match["relationship"] == "false_friend"
    assert match["gloss_confidence"] == "dataset"
    assert "boat" in match["other_gloss"].lower() or "vessel" in match["other_gloss"].lower()


def test_reverse_lookup_empty_for_unknown_word() -> None:
    assert reverse_lookup("zzzqxnotaword999𐀀", "kn", model=load_model()) == []
    assert reverse_lookup("zzzqxnotaword999𐀀", "te", model=load_model()) == []


def test_reverse_lookup_dedup_prefers_gold() -> None:
    """ತರಿ/తరి is in gold and stream_b; gold verified row must win."""
    hits = reverse_lookup("ತರಿ", "kn", model=load_model())
    same_te = [h for h in hits if h["other_word"] == "తరి"]
    assert len(same_te) == 1
    assert same_te[0]["source"] == "verified"
    assert same_te[0]["relationship"] == "false_friend"


def _empty_lookup_tables(stream_row: dict) -> dict:
    cols = ["kn_word", "te_word", "en_kn", "en_te", "kn_iso", "te_iso"]
    gold_cols = cols + ["final_label", "pair_id"]
    return {
        "gold": pd.DataFrame(columns=gold_cols),
        "stream_a": pd.DataFrame([stream_row]),
        "stream_b": pd.DataFrame(columns=cols),
        "features": pd.DataFrame(),
    }


def test_reverse_lookup_gloss_backfill_sets_lookup_confidence(monkeypatch) -> None:
    tables = _empty_lookup_tables(
        {
            "kn_word": "ಕನ್ನಡಪದ",
            "te_word": "తెలుగుపద",
            "en_kn": "kannada meaning",
            "en_te": "",
            "kn_iso": "x",
            "te_iso": "y",
        }
    )
    monkeypatch.setattr("demo_utils._load_lookup_tables", lambda: tables)
    monkeypatch.setattr(
        "gloss_lookup.find_gloss",
        lambda word, lang: "backfilled gloss" if word == "తెలుగుపద" else None,
    )
    monkeypatch.setattr(
        "demo_utils.check_word_pair",
        lambda *a, **k: {
            "predicted_label": "unrelated",
            "confidence": 0.9,
            "low_confidence": False,
            "sem_sim": 0.4,
        },
    )
    hits = reverse_lookup("ಕನ್ನಡಪದ", "kn", model=MagicMock())
    assert len(hits) == 1
    assert hits[0]["other_word"] == "తెలుగుపద"
    assert hits[0]["other_gloss"] == "backfilled gloss"
    assert hits[0]["gloss_confidence"] == "lookup"


def test_reverse_lookup_gloss_backfill_unavailable_when_lookup_fails(monkeypatch) -> None:
    tables = _empty_lookup_tables(
        {
            "kn_word": "ಕನ್ನಡಪದ",
            "te_word": "తెలుగుపద",
            "en_kn": "kannada meaning",
            "en_te": "",
            "kn_iso": "x",
            "te_iso": "y",
        }
    )
    monkeypatch.setattr("demo_utils._load_lookup_tables", lambda: tables)
    monkeypatch.setattr("gloss_lookup.find_gloss", lambda word, lang: None)
    monkeypatch.setattr(
        "demo_utils.check_word_pair",
        lambda *a, **k: {
            "predicted_label": "unrelated",
            "confidence": 0.9,
            "low_confidence": True,
            "sem_sim": 0.5,
        },
    )
    hits = reverse_lookup("ಕನ್ನಡಪದ", "kn", model=MagicMock())
    assert len(hits) == 1
    assert hits[0]["other_gloss"] is None
    assert hits[0]["gloss_confidence"] == "unavailable"
