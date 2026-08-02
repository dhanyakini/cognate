"""Tests for English glossing (faithful, never fabricated)."""

from __future__ import annotations

from cognate.glossing import gloss_row, gloss_rows, split_bilingual_gloss


def test_split_bilingual_gloss() -> None:
    kn, te = split_bilingual_gloss("kn: ನೀರು || te: నీరు")
    assert kn == "ನೀರು"
    assert te == "నీరు"


def test_no_english_source_leaves_blank_and_needs_gloss() -> None:
    row = {
        "pair_id": "A1",
        "kn_word": "ನೀರು",
        "te_word": "నీరు",
        "synset_id": "",
        "gloss": "kn: ನೀರು || te: నీరు",
    }

    def no_wn(sid: str, native: str) -> str | None:
        return None

    def no_mt(text: str) -> str | None:
        return None

    out = gloss_row(row, wordnet_fn=no_wn, translate_fn=no_mt)
    assert out["en_kn"] == ""
    assert out["en_te"] == ""
    assert out["needs_gloss"] == "true"


def test_wordnet_linked_row_gets_nonempty_en() -> None:
    row = {
        "pair_id": "A1",
        "synset_id": "42",
        "gloss": "kn: ನೀರು || te: నీరు",
    }

    def fake_wn(sid: str, native: str) -> str | None:
        return "water" if sid == "42" else None

    out = gloss_row(row, wordnet_fn=fake_wn, translate_fn=lambda t: None)
    assert out["en_kn"] == "water"
    assert out["en_te"] == "water"
    assert out["needs_gloss"] == "false"


def test_lemma_fallback_glosses_random_pairs_without_bilingual_gloss() -> None:
    row = {
        "pair_id": "R1",
        "kn_word": "ನೀರು",
        "te_word": "నీరు",
        "synset_id": "",
        "gloss": "",
    }

    def fake_lemma(word: str, lang: str) -> str:
        return {"kn": "ನೀರು", "te": "నీరు"}[lang] if word else ""

    def mt(text: str) -> str | None:
        return f"EN:{text}" if text.strip() else None

    out = gloss_row(
        row,
        wordnet_fn=lambda s, n: None,
        translate_fn=mt,
        lemma_fn=fake_lemma,
    )
    assert out["en_kn"] == "EN:ನೀರು"
    assert out["en_te"] == "EN:నీరు"
    assert out["needs_gloss"] == "false"


def test_mt_path_counts_and_never_invents_without_native() -> None:
    rows = [
        {"pair_id": "B1", "synset_id": "", "gloss": "kn: || te: "},
        {"pair_id": "B2", "synset_id": "", "gloss": "kn: ಹಕ್ಕಿ || te: పక్షి"},
    ]

    def mt(text: str) -> str | None:
        return f"EN:{text}" if text.strip() else None

    glossed, stats = gloss_rows(
        rows, wordnet_fn=lambda s, n: None, translate_fn=mt
    )
    assert glossed[0]["needs_gloss"] == "true"
    assert glossed[1]["en_kn"] == "EN:ಹಕ್ಕಿ"
    assert glossed[1]["en_te"] == "EN:పక్షి"
    assert stats.glossed_via_mt == 2
    assert stats.needs_gloss == 1
