"""
Phonetic / sound-class similarity on ISO-15919 romanizations.

Two backends share the same [0, 1] normalization:

    sim(a, b) = clamp(0, 1, S(a, b) / S(L, L))

where S is the raw global-alignment similarity score and L is the longer
of the two tokenized strings (ties: either). S(L, L) is the self-alignment
score of L under the same scorer — the maximum achievable score for a
string of that length when every position is an identity match. Empty /
empty → 1.0; exactly one empty → 0.0.

Primary backend ("lingpy"): LingPy ``Pairwise`` with ``model="sca"`` and
``mode="global"`` (SCA sound-class scorer; see List 2012). Optional dep:
``pip install -e ".[phonetic]"``.

Fallback backend ("nw"): Needleman–Wunsch over ISO phone tokens with a
feature-based substitution matrix (place / manner / voicing for
consonants; height / backness / length for vowels).
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import MutableMapping

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-based substitution matrix (NW backend)
# ---------------------------------------------------------------------------
# Scores are similarities in [GAP_PENALTY, IDENTITY_SCORE]. Distance on each
# feature axis is absolute difference of integer codes; total feature
# distance is summed then mapped to a substitution score. Swap or extend
# PHONE_FEATURES / FEATURE_WEIGHTS without touching the aligner.

IDENTITY_SCORE = 2.0
GAP_PENALTY = -1.0
MIN_SUBSTITUTION = -1.0

# feature_name -> weight in the distance sum
FEATURE_WEIGHTS: dict[str, float] = {
    "place": 1.0,
    "manner": 1.0,
    "voice": 0.5,
    "height": 1.0,
    "backness": 1.0,
    "length": 0.5,
}

# phone (NFC, lower) -> feature vector. Unknown phones fall back to a
# neutral "other" consonant or vowel guess from character class.
# place: 0 bilabial … 5 glottal
# manner: 0 stop, 1 affricate, 2 fricative, 3 nasal, 4 liquid, 5 glide
# voice: 0 voiceless, 1 voiced
# height: 0 high … 3 low; backness: 0 front … 2 back; length: 0 short, 1 long
PHONE_FEATURES: dict[str, dict[str, float]] = {
    # vowels
    "a": {"height": 3, "backness": 1, "length": 0},
    "ā": {"height": 3, "backness": 1, "length": 1},
    "i": {"height": 0, "backness": 0, "length": 0},
    "ī": {"height": 0, "backness": 0, "length": 1},
    "u": {"height": 0, "backness": 2, "length": 0},
    "ū": {"height": 0, "backness": 2, "length": 1},
    "e": {"height": 1, "backness": 0, "length": 0},
    "ē": {"height": 1, "backness": 0, "length": 1},
    "o": {"height": 1, "backness": 2, "length": 0},
    "ō": {"height": 1, "backness": 2, "length": 1},
    "ai": {"height": 2, "backness": 0, "length": 0},
    "au": {"height": 2, "backness": 2, "length": 0},
    # labials
    "p": {"place": 0, "manner": 0, "voice": 0},
    "ph": {"place": 0, "manner": 0, "voice": 0},
    "b": {"place": 0, "manner": 0, "voice": 1},
    "bh": {"place": 0, "manner": 0, "voice": 1},
    "m": {"place": 0, "manner": 3, "voice": 1},
    "v": {"place": 0, "manner": 5, "voice": 1},
    # dentals / alveolars
    "t": {"place": 1, "manner": 0, "voice": 0},
    "th": {"place": 1, "manner": 0, "voice": 0},
    "d": {"place": 1, "manner": 0, "voice": 1},
    "dh": {"place": 1, "manner": 0, "voice": 1},
    "n": {"place": 1, "manner": 3, "voice": 1},
    "s": {"place": 1, "manner": 2, "voice": 0},
    "l": {"place": 1, "manner": 4, "voice": 1},
    "r": {"place": 1, "manner": 4, "voice": 1},
    # retroflex
    "ṭ": {"place": 2, "manner": 0, "voice": 0},
    "ṭh": {"place": 2, "manner": 0, "voice": 0},
    "ḍ": {"place": 2, "manner": 0, "voice": 1},
    "ḍh": {"place": 2, "manner": 0, "voice": 1},
    "ṇ": {"place": 2, "manner": 3, "voice": 1},
    "ṣ": {"place": 2, "manner": 2, "voice": 0},
    "ḷ": {"place": 2, "manner": 4, "voice": 1},
    # palatal
    "c": {"place": 3, "manner": 1, "voice": 0},
    "ch": {"place": 3, "manner": 1, "voice": 0},
    "j": {"place": 3, "manner": 1, "voice": 1},
    "jh": {"place": 3, "manner": 1, "voice": 1},
    "ñ": {"place": 3, "manner": 3, "voice": 1},
    "ś": {"place": 3, "manner": 2, "voice": 0},
    "y": {"place": 3, "manner": 5, "voice": 1},
    # velar
    "k": {"place": 4, "manner": 0, "voice": 0},
    "kh": {"place": 4, "manner": 0, "voice": 0},
    "g": {"place": 4, "manner": 0, "voice": 1},
    "gh": {"place": 4, "manner": 0, "voice": 1},
    "ṅ": {"place": 4, "manner": 3, "voice": 1},
    "h": {"place": 5, "manner": 2, "voice": 0},
    # anusvāra / visarga / candrabindu-ish
    "ṁ": {"place": 4, "manner": 3, "voice": 1},
    "ṃ": {"place": 4, "manner": 3, "voice": 1},
    "ḥ": {"place": 5, "manner": 2, "voice": 0},
}

_ASPIRATE_DIGRAPHS = (
    "kh",
    "gh",
    "ch",
    "jh",
    "ṭh",
    "ḍh",
    "th",
    "dh",
    "ph",
    "bh",
)
_VOWEL_DIGRAPHS = ("ai", "au")
_VOWEL_CHARS = set("aāiīuūeēoō")


def tokenize_iso(iso: str) -> list[str]:
    """Split an ISO-15919 string into phone-like tokens (NFC, lowercased)."""
    s = unicodedata.normalize("NFC", (iso or "").strip().lower())
    if not s:
        return []
    tokens: list[str] = []
    i = 0
    while i < len(s):
        if i + 1 < len(s):
            digraph = s[i : i + 2]
            if digraph in _ASPIRATE_DIGRAPHS or digraph in _VOWEL_DIGRAPHS:
                tokens.append(digraph)
                i += 2
                continue
        ch = s[i]
        if ch.isalpha() or ch in {"ṁ", "ṃ", "ḥ", "ṅ", "ñ", "ṇ", "ṭ", "ḍ", "ṣ", "ś", "ḷ"}:
            tokens.append(ch)
        # skip punctuation / whitespace / hyphens
        i += 1
    return tokens


def _is_vowel_phone(phone: str) -> bool:
    if phone in PHONE_FEATURES:
        return "height" in PHONE_FEATURES[phone]
    return bool(phone) and phone[0] in _VOWEL_CHARS


def phone_features(phone: str) -> dict[str, float]:
    """Return a feature vector for ``phone``, with a conservative fallback."""
    if phone in PHONE_FEATURES:
        return PHONE_FEATURES[phone]
    if _is_vowel_phone(phone):
        return {"height": 2, "backness": 1, "length": 0}
    return {"place": 3, "manner": 0, "voice": 0}


def substitution_score(a: str, b: str) -> float:
    """Feature-distance substitution score for two phones."""
    if a == b:
        return IDENTITY_SCORE
    fa, fb = phone_features(a), phone_features(b)
    # Vowel–consonant mismatch: strong penalty.
    if ("height" in fa) != ("height" in fb):
        return MIN_SUBSTITUTION
    keys = set(fa) & set(fb)
    dist = 0.0
    for key in keys:
        weight = FEATURE_WEIGHTS.get(key, 1.0)
        dist += weight * abs(fa[key] - fb[key])
    # Map distance → score; distance 0 → IDENTITY, large → MIN_SUBSTITUTION.
    score = IDENTITY_SCORE - dist
    return max(MIN_SUBSTITUTION, score)


def nw_alignment_score(a_tokens: list[str], b_tokens: list[str]) -> float:
    """Needleman–Wunsch global alignment score (raw, not normalized)."""
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 and m == 0:
        return 0.0
    # DP: prev/curr rows
    prev = [GAP_PENALTY * j for j in range(m + 1)]
    for i in range(1, n + 1):
        curr = [GAP_PENALTY * i]
        ca = a_tokens[i - 1]
        for j in range(1, m + 1):
            cb = b_tokens[j - 1]
            match = prev[j - 1] + substitution_score(ca, cb)
            delete = prev[j] + GAP_PENALTY
            insert = curr[j - 1] + GAP_PENALTY
            curr.append(max(match, delete, insert))
        prev = curr
    return float(prev[-1])


def _normalize_by_longer_self(
    raw: float,
    a_tokens: list[str],
    b_tokens: list[str],
    self_score_fn,
) -> float:
    """``clamp(0, 1, raw / S(L, L))`` with L = longer token sequence."""
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    longer = a_tokens if len(a_tokens) >= len(b_tokens) else b_tokens
    denom = float(self_score_fn(longer, longer))
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / denom))


def nw_similarity(a_iso: str, b_iso: str) -> float:
    """Hand-rolled NW phonetic similarity in [0, 1] (see module docstring)."""
    a_toks = tokenize_iso(a_iso)
    b_toks = tokenize_iso(b_iso)
    raw = nw_alignment_score(a_toks, b_toks)
    return _normalize_by_longer_self(raw, a_toks, b_toks, nw_alignment_score)


def _lingpy_raw_score(a_iso: str, b_iso: str) -> float:
    """Raw SCA global-alignment similarity via LingPy ``Pairwise``."""
    from lingpy.align.pairwise import Pairwise

    pw = Pairwise(a_iso, b_iso)
    pw.align(model="sca", mode="global", distance=False)
    return float(pw.alignments[0][-1])


def sca_similarity(a_iso: str, b_iso: str) -> float:
    """
    LingPy SCA pairwise similarity, normalized to [0, 1].

    Uses ``lingpy.align.pairwise.Pairwise`` with ``model="sca"`` and
    ``mode="global"`` (returns the raw SCA similarity score, not Downey
    distance). Normalization:

        sim = clamp(0, 1, S(a, b) / S(L, L))

    where ``L`` is the longer of the two *character* strings (LingPy
    tokenizes internally; we use string length only to pick which input
    is self-aligned for the denominator). Empty/empty → 1.0; one empty → 0.0.
    """
    a = (a_iso or "").strip()
    b = (b_iso or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    raw = _lingpy_raw_score(a, b)
    longer = a if len(a) >= len(b) else b
    denom = _lingpy_raw_score(longer, longer)
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / denom))


def _resolve_backend(backend: str) -> str:
    backend = (backend or "auto").strip().lower()
    if backend not in {"auto", "lingpy", "nw"}:
        raise ValueError(f"unknown phonetic backend: {backend!r}")
    if backend == "lingpy":
        return "lingpy"
    if backend == "nw":
        return "nw"
    try:
        import lingpy  # noqa: F401
    except ImportError:
        log.info("phonetic backend=nw (lingpy not importable)")
        return "nw"
    log.info("phonetic backend=lingpy")
    return "lingpy"


def phonetic_similarity(
    a_iso: str,
    b_iso: str,
    backend: str = "auto",
) -> float:
    """
    Phonetic similarity in [0, 1] between two ISO-15919 strings.

    ``backend``: ``"lingpy"``, ``"nw"``, or ``"auto"`` (try LingPy, else NW).
    Logs which backend actually ran.
    """
    chosen = _resolve_backend(backend)
    if chosen == "lingpy":
        if backend == "lingpy":
            log.info("phonetic backend=lingpy")
        return sca_similarity(a_iso, b_iso)
    if backend == "nw":
        log.info("phonetic backend=nw")
    return nw_similarity(a_iso, b_iso)


# ---------------------------------------------------------------------------
# Disk cache (same pattern as cognate.transliterate)
# ---------------------------------------------------------------------------

def cache_key(backend: str, a_iso: str, b_iso: str) -> str:
    """Canonical key so sim(a,b) and sim(b,a) share one cache entry."""
    left, right = sorted((a_iso or "", b_iso or ""))
    return f"{backend}\t{left}\t{right}"


def load_cache(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def save_cache(cache: MutableMapping[str, float], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(cache), f, ensure_ascii=False, indent=0)
