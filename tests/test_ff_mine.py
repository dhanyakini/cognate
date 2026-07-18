"""Tests for Stream B false-friend mining filters."""

from __future__ import annotations

from cognate.ff_mine import LemmaRecord, mine_pairs, should_keep_pair
from cognate.features.orthographic import normalized_similarity


def _rec(word: str, iso: str, synsets: set[str], gloss: str = "") -> LemmaRecord:
    return LemmaRecord(
        word=word,
        iso=iso,
        synsets=frozenset(synsets),
        gloss=gloss,
    )


def test_form_similar_different_synset_kept() -> None:
    kn = _rec("ಕಾ", "kaa", {"1"}, "kn gloss")
    te = _rec("కా", "kaa", {"99"}, "te gloss")
    sim = normalized_similarity(kn.iso, te.iso)
    assert sim >= 0.75
    assert should_keep_pair(kn, te, sim, threshold=0.75)

    kept = mine_pairs([kn], [te], threshold=0.75, length_tolerance=2, max_pairs=3000)
    assert len(kept) == 1
    assert kept[0].kn_word == kn.word
    assert kept[0].te_word == te.word
    assert "kn:" in kept[0].gloss and "te:" in kept[0].gloss


def test_form_similar_same_synset_excluded() -> None:
    kn = _rec("ನೀರು", "niiru", {"42", "7"})
    te = _rec("నీరు", "niiru", {"42", "100"})
    sim = normalized_similarity(kn.iso, te.iso)
    assert sim >= 0.75
    assert not should_keep_pair(kn, te, sim, threshold=0.75)

    kept = mine_pairs([kn], [te], threshold=0.75, length_tolerance=2, max_pairs=3000)
    assert kept == []


def test_dissimilar_pair_excluded() -> None:
    kn = _rec("ಮನೆ", "mane", {"1"})
    te = _rec("పుస్తకం", "pustakam", {"2"})
    sim = normalized_similarity(kn.iso, te.iso)
    assert sim < 0.75
    assert not should_keep_pair(kn, te, sim, threshold=0.75)

    kept = mine_pairs([kn], [te], threshold=0.75, length_tolerance=2, max_pairs=3000)
    assert kept == []
