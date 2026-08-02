"""Tests for English-gloss semantic similarity (encoder always mocked)."""

from __future__ import annotations

from typing import Any

import pytest

from cognate.features import semantic as sem


class FakeEncoder:
    """Deterministic mock: known strings → fixed vectors; else call-counted."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def encode(self, sentences: str | list[str], **kwargs: Any) -> list[float]:
        text = sentences if isinstance(sentences, str) else sentences[0]
        self.calls.append(text)
        if text not in self.mapping:
            raise KeyError(f"no fake embedding for {text!r}")
        return list(self.mapping[text])


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    sem.reset_encoder()
    yield
    sem.reset_encoder()


def test_identical_strings_similarity_near_one() -> None:
    enc = FakeEncoder({"water": [1.0, 0.0, 0.0]})
    cache: dict[str, list[float]] = {}
    score = sem.semantic_similarity("water", "water", cache=cache, encoder=enc)
    assert score is not None
    assert score == pytest.approx(1.0)


def test_orthogonal_vectors_map_to_midpoint() -> None:
    """cos=0 → (0+1)/2 = 0.5 — checks the mapping, not LaBSE."""
    enc = FakeEncoder(
        {
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        }
    )
    score = sem.semantic_similarity("a", "b", cache={}, encoder=enc)
    assert score == pytest.approx(0.5)


def test_blank_gloss_returns_none_without_calling_encoder() -> None:
    enc = FakeEncoder({"x": [1.0]})
    assert sem.semantic_similarity("", "hello", encoder=enc) is None
    assert sem.semantic_similarity("hello", "   ", encoder=enc) is None
    assert sem.semantic_similarity(None, "hello", encoder=enc) is None  # type: ignore[arg-type]
    assert enc.calls == []


def test_cache_avoids_reencoding_same_gloss() -> None:
    enc = FakeEncoder(
        {
            "mile (unit of distance)": [1.0, 0.0],
            "mile (5,280 feet)": [0.9, 0.1],
        }
    )
    cache: dict[str, list[float]] = {}
    s1 = sem.semantic_similarity(
        "mile (unit of distance)",
        "mile (5,280 feet)",
        cache=cache,
        encoder=enc,
    )
    assert s1 is not None
    assert len(enc.calls) == 2

    s2 = sem.semantic_similarity(
        "mile (unit of distance)",
        "mile (5,280 feet)",
        cache=cache,
        encoder=enc,
    )
    assert s2 == s1
    assert len(enc.calls) == 2  # no further encode calls

    # Same gloss on a different pair still hits cache.
    s3 = sem.semantic_similarity(
        "mile (unit of distance)",
        "mile (unit of distance)",
        cache=cache,
        encoder=enc,
    )
    assert s3 == pytest.approx(1.0)
    assert len(enc.calls) == 2


def test_negative_cosine_is_mapped_not_clamped() -> None:
    enc = FakeEncoder(
        {
            "anti_a": [1.0, 0.0],
            "anti_b": [-1.0, 0.0],
        }
    )
    score = sem.semantic_similarity("anti_a", "anti_b", cache={}, encoder=enc)
    assert score == pytest.approx(0.0)  # ( -1 + 1 ) / 2
