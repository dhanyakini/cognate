"""Tests for cognate.normalize."""

from __future__ import annotations

import csv
from pathlib import Path

from cognate import CSV_HEADER
from cognate.normalize import (
    NormalizeConfig,
    clean_word,
    normalize_rows,
    write_clean_csv,
)


def _raw(
    kn: str,
    te: str,
    *,
    synset_id: str = "1",
    gloss_en: str = "ಕನ್ನಡ ಗ್ಲಾಸ್",
    source: str = "shared_synset",
) -> dict[str, str]:
    return {
        "pair_id": "",
        "kn_word": kn,
        "te_word": te,
        "kn_iso": "STALE",
        "te_iso": "STALE",
        "synset_id": synset_id,
        "gloss_en": gloss_en,
        "candidate_source": source,
        "label": "x",
        "origin": "y",
        "annotator": "z",
        "notes": "n",
    }


def test_clean_word_strips_trailing_period() -> None:
    assert clean_word("గొంతు.") == "గొంతు"
    assert clean_word("  mane,  ") == "mane"


def test_normalize_drops_multiword_and_digit_and_len1(tmp_path: Path) -> None:
    rows = [
        _raw("ನೀರು_ಮನೆ", "నీరు"),  # multiword
        _raw("ನೀರು", "నీ_రు"),  # multiword te
        _raw("ಅ", "అ"),  # len 1
        _raw("12", "నీరు"),  # digit
        _raw("ನೀರು", "౧౨"),  # telugu digit
        _raw(".", "నీరు"),  # empty after clean
    ]
    cfg = NormalizeConfig(
        keep_multiword=False,
        translit_cache_path=str(tmp_path / "cache.json"),
        te_gloss_by_id={"1": "తెలుగు గ్లాస్"},
    )
    cleaned, stats = normalize_rows(rows, "a", cfg)
    assert cleaned == []
    assert stats.dropped_multiword >= 2
    assert stats.dropped_numeric_or_len1 >= 3
    assert stats.dropped_empty_after_clean >= 1


def test_normalize_cleans_period_recomputes_iso_dedupes_and_gloss(tmp_path: Path) -> None:
    rows = [
        _raw("ನೀರು", "నీరు.", gloss_en="ಯಾವುದು ನೀರು", synset_id="42"),
        _raw("ನೀರು", "నీరు.", gloss_en="duplicate sense", synset_id="99"),  # dup pair
        _raw("ಮನೆ", "ఇల్లు", gloss_en="ಮನೆ ಎಂದರೆ", synset_id="7"),
    ]
    cfg = NormalizeConfig(
        keep_multiword=False,
        translit_cache_path=str(tmp_path / "cache.json"),
        te_gloss_by_id={"42": "నీరు అనగా", "7": "ఇల్లు అనగా"},
    )
    cleaned, stats = normalize_rows(rows, "a", cfg)
    assert stats.deduped == 1
    assert len(cleaned) == 2

    first = cleaned[0]
    assert first["te_word"] == "నీరు"  # period stripped
    assert first["kn_iso"]
    assert first["te_iso"]
    assert first["kn_iso"] != "STALE"
    assert "." not in first["te_iso"]
    assert "gloss" in first and "gloss_en" not in first
    assert first["gloss"].startswith("kn:")
    assert "||" in first["gloss"]
    assert "te:" in first["gloss"]
    assert "నీరు అనగా" in first["gloss"]
    assert first["label"] == ""

    out = tmp_path / "clean.csv"
    write_clean_csv(cleaned, out)
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_HEADER
        written = list(reader)
    assert all(r["kn_iso"] and r["te_iso"] for r in written)
    assert all(r["gloss"].startswith("kn:") and "||" in r["gloss"] for r in written)


def test_normalize_stream_b_renames_gloss(tmp_path: Path) -> None:
    rows = [
        {
            "pair_id": "B1",
            "kn_word": "ಕಾಗೆ",
            "te_word": "కాకి",
            "kn_iso": "old",
            "te_iso": "old",
            "synset_id": "",
            "gloss_en": "kn: ಹಕ್ಕಿ || te: పక్షి",
            "candidate_source": "form_similar",
            "label": "",
            "origin": "",
            "annotator": "",
            "notes": "",
        }
    ]
    cfg = NormalizeConfig(
        translit_cache_path=str(tmp_path / "cache.json"),
        te_gloss_by_id={},
    )
    cleaned, _ = normalize_rows(rows, "b", cfg)
    assert len(cleaned) == 1
    assert cleaned[0]["gloss"] == "kn: ಹಕ್ಕಿ || te: పక్షి"
    assert cleaned[0]["synset_id"] == ""
    assert cleaned[0]["kn_iso"] != "old"


def test_normalize_drops_internal_punct_and_overlong(tmp_path: Path) -> None:
    rows = [
        _raw("ನೀ.ರು", "నీరు"),  # internal period
        _raw("ನೀರು", 'నీ"రు'),  # embedded quote
        _raw("ನೀರು", "నీరు" + "x" * 50),  # overlong te
        _raw("ನೀರು", "నీరు"),  # keep
    ]
    cfg = NormalizeConfig(
        keep_multiword=False,
        max_len=40,
        translit_cache_path=str(tmp_path / "cache.json"),
        te_gloss_by_id={"1": "తెలుగు"},
    )
    cleaned, stats = normalize_rows(rows, "a", cfg)
    assert len(cleaned) == 1
    assert cleaned[0]["kn_word"] == "ನೀರು"
    assert stats.dropped_internal_punct >= 2
    assert stats.dropped_overlong >= 1


def test_streams_disjoint_helper() -> None:
    from cognate.normalize import assert_streams_disjoint, drop_pairs_in_other, pair_keys

    a = [{"kn_word": "ನೀರು", "te_word": "నీరు"}, {"kn_word": "ಮನೆ", "te_word": "ఇల్లు"}]
    b = [{"kn_word": "ನೀರು", "te_word": "నీరు"}, {"kn_word": "ಕಾಗೆ", "te_word": "కాకి"}]
    filtered = drop_pairs_in_other(b, pair_keys(a))
    assert_streams_disjoint(a, filtered)
    assert len(filtered) == 1
    assert filtered[0]["kn_word"] == "ಕಾಗೆ"
