"""Tests for phonetic / sound-class similarity."""

from __future__ import annotations

import builtins
import importlib.util
from unittest.mock import patch

import pytest

from cognate.features.phonetic import (
    nw_similarity,
    phonetic_similarity,
    sca_similarity,
    tokenize_iso,
)

# Pilot pairs from data/pilot_labeled_dhanya.csv
COGNATE_PAIRS = [
    ("sādhana", "sādhanaṁ"),  # P000002 cognate
    ("maili", "mailu"),  # P000007 cognate
    ("parōpakāra", "parōpakāramu"),  # P000006 cognate
]
UNRELATED_PAIR = ("mette", "auṣṭhyālu")  # P000001 unrelated


def test_identical_strings_score_one() -> None:
    assert nw_similarity("sādhana", "sādhana") == 1.0
    assert phonetic_similarity("mailu", "mailu", backend="nw") == 1.0
    assert nw_similarity("", "") == 1.0


def test_cognates_score_higher_than_unrelated() -> None:
    unrelated = nw_similarity(*UNRELATED_PAIR)
    for kn, te in COGNATE_PAIRS:
        cognate = nw_similarity(kn, te)
        assert cognate > unrelated, (kn, te, cognate, unrelated)


def test_auto_falls_back_to_nw_when_lingpy_missing() -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lingpy" or name.startswith("lingpy."):
            raise ImportError("mocked missing lingpy")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        score = phonetic_similarity("maili", "mailu", backend="auto")
    assert score == nw_similarity("maili", "mailu")
    assert 0.0 < score <= 1.0


def test_nw_is_deterministic_and_symmetric() -> None:
    a, b = "guṇagāna", "guṇagānaṁ"
    s1 = nw_similarity(a, b)
    s2 = nw_similarity(a, b)
    s3 = nw_similarity(b, a)
    assert s1 == s2 == s3
    assert 0.0 <= s1 <= 1.0


def test_tokenize_iso_keeps_aspirates_and_diacritics() -> None:
    assert "dh" in tokenize_iso("sādhana")
    assert "ā" in tokenize_iso("sādhana")
    assert "ṭh" in tokenize_iso("auṣṭhyālu")


@pytest.mark.skipif(
    importlib.util.find_spec("lingpy") is None,
    reason="lingpy optional dep not installed",
)
def test_sca_similarity_available_and_bounded() -> None:
    score = sca_similarity("sādhana", "sādhanaṁ")
    assert 0.0 <= score <= 1.0
    assert score > sca_similarity(*UNRELATED_PAIR)
    assert sca_similarity("x", "x") == 1.0
