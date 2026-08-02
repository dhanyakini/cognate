"""
Semantic similarity from English glosses (en_kn / en_te).

Uses multilingual sentence embeddings (LaBSE via sentence-transformers) to
score whether the two sides of a pair mean the same thing. Input is always
the English glosses — never native script or ISO-15919 forms.

Cosine mapping (do not change without updating tests):

    sim = (cos(u, v) + 1) / 2

This maps LaBSE cosine from [-1, 1] onto [0, 1] without discarding negative
cosines (clamping to 0 would treat "anti-aligned" the same as orthogonal).

Missing glosses: if either side is blank, return ``None`` (not 0.0). A bare
0.0 would be indistinguishable from "confidently dissimilar" and would
corrupt ablations that impute or drop missing features.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, MutableMapping, Protocol, Sequence

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/LaBSE"

_encoder: Any | None = None


class Encoder(Protocol):
    """Minimal encode interface (real SentenceTransformer or a test mock)."""

    def encode(self, sentences: str | list[str], **kwargs: Any) -> Any: ...


def get_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    """Lazy singleton SentenceTransformer loader (never runs at import time)."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        log.info("loading semantic encoder %s", model_name)
        _encoder = SentenceTransformer(model_name)
    return _encoder


def reset_encoder() -> None:
    """Drop the singleton (tests / model swap)."""
    global _encoder
    _encoder = None


def gloss_is_missing(text: str | None) -> bool:
    """True when a gloss cannot support a semantic score."""
    return text is None or not str(text).strip()


def content_hash(gloss: str) -> str:
    """Stable cache key for an exact gloss string."""
    return hashlib.sha256(gloss.encode("utf-8")).hexdigest()


def cosine_similarity(u: Sequence[float], v: Sequence[float]) -> float:
    """Raw cosine in [-1, 1]."""
    if len(u) != len(v) or not u:
        raise ValueError("embedding length mismatch or empty")
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v, strict=True):
        fa, fb = float(a), float(b)
        dot += fa * fb
        nu += fa * fa
        nv += fb * fb
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def map_cosine_to_unit(cos: float) -> float:
    """Map cosine from [-1, 1] to [0, 1] via ``(cos + 1) / 2``."""
    return (float(cos) + 1.0) / 2.0


def _as_float_list(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(x) for x in vector]


def embed_gloss(
    gloss: str,
    cache: MutableMapping[str, list[float]],
    encoder: Encoder | None = None,
) -> list[float]:
    """
    Return the embedding for ``gloss``, using ``cache`` keyed by content hash.

    Identical gloss strings across pairs reuse one stored vector.
    """
    key = content_hash(gloss)
    if key in cache:
        return cache[key]
    enc = encoder if encoder is not None else get_encoder()
    vector = _as_float_list(enc.encode(gloss))
    cache[key] = vector
    return vector


def semantic_similarity(
    en_a: str | None,
    en_b: str | None,
    *,
    cache: MutableMapping[str, list[float]] | None = None,
    encoder: Encoder | None = None,
) -> float | None:
    """
    Meaning similarity of two English glosses in [0, 1], or ``None`` if either
    gloss is blank.

    ``sim = (cosine(embed(en_a), embed(en_b)) + 1) / 2``

    Embeddings are cached by content hash of the exact gloss string when a
    ``cache`` mapping is provided (see ``load_cache`` / ``save_cache`` and
    ``semantic.cache_path`` in config.yaml).
    """
    if gloss_is_missing(en_a) or gloss_is_missing(en_b):
        return None
    store: MutableMapping[str, list[float]] = cache if cache is not None else {}
    u = embed_gloss(str(en_a).strip(), store, encoder=encoder)
    # Re-strip consistently with the blank check, but cache the stripped form.
    v = embed_gloss(str(en_b).strip(), store, encoder=encoder)
    return map_cosine_to_unit(cosine_similarity(u, v))


# ---------------------------------------------------------------------------
# Disk cache (same pattern as cognate.transliterate / features.phonetic)
# ---------------------------------------------------------------------------

def load_cache(path: str | Path) -> dict[str, list[float]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[float]] = {}
    for k, v in data.items():
        if isinstance(v, list) and v:
            try:
                out[str(k)] = [float(x) for x in v]
            except (TypeError, ValueError):
                continue
    return out


def save_cache(cache: MutableMapping[str, list[float]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({k: list(v) for k, v in cache.items()}, f, ensure_ascii=False)
