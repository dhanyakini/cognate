"""Tests for gloss_lookup and check_word_pair."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from demo_utils import check_word_pair, load_model
from gloss_lookup import find_gloss, get_gloss_or_prompt, reset_gloss_caches


def setup_function() -> None:
    reset_gloss_caches()


def test_find_gloss_known_gold_word() -> None:
    gold = pd.read_csv("data/gold.csv")
    row = gold.iloc[0]
    kn_word = str(row["kn_word"])
    expected = str(row["en_kn"]).strip()
    assert expected
    got = find_gloss(kn_word, "kn")
    assert got == expected
    gloss, auto = get_gloss_or_prompt(kn_word, "kn")
    assert gloss == expected
    assert auto is True


def test_find_gloss_nonsense_returns_none() -> None:
    nonsense = "zzzqxnotaword999𐀀"
    assert find_gloss(nonsense, "kn") is None
    assert find_gloss(nonsense, "te") is None
    gloss, auto = get_gloss_or_prompt(nonsense, "kn")
    assert gloss is None
    assert auto is False


def test_check_word_pair_low_confidence_when_gloss_missing() -> None:
    model = load_model()
    with patch("cognate.features.semantic.semantic_similarity") as sem:
        result = check_word_pair("ಸಾಧನ", "సాధనం", None, "tool", model)
        sem.assert_not_called()
    assert result["low_confidence"] is True
    assert result["sem_sim"] == 0.5
    assert result["predicted_label"] in {"cognate", "false_friend", "unrelated"}


def test_check_word_pair_low_confidence_false_when_both_glosses() -> None:
    model = load_model()
    with patch(
        "cognate.features.semantic.semantic_similarity",
        return_value=0.91,
    ) as sem:
        result = check_word_pair(
            "ಸಾಧನ",
            "సాధనం",
            "tool / instrument",
            "tool / instrument",
            model,
        )
        sem.assert_called_once()
    assert result["low_confidence"] is False
    assert result["sem_sim"] == 0.91
    assert result["kn_gloss_used"] == "tool / instrument"
    assert result["te_gloss_used"] == "tool / instrument"
