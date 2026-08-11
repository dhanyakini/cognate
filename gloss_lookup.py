"""
gloss_lookup.py — tiered English/synset gloss lookup for live word checks.

Order: gold.csv → stream_a_glossed.csv → stream_b_glossed.csv → IndoWordNet.
CSV indexes and IWN language objects are lazy-loaded once and cached in memory.
IndoWordNet data is local (pyiwn bundle); no network calls after install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold.csv"
STREAM_A_GLOSSED = ROOT / "data" / "candidates" / "stream_a_glossed.csv"
STREAM_B_GLOSSED = ROOT / "data" / "candidates" / "stream_b_glossed.csv"
_CSV_SOURCES = (GOLD_PATH, STREAM_A_GLOSSED, STREAM_B_GLOSSED)

# Module-level caches (lazy).
_CSV_KN: dict[str, str] | None = None
_CSV_TE: dict[str, str] | None = None
_IWN_BY_LANG: dict[str, Any] = {}


def reset_gloss_caches() -> None:
    """Clear CSV / IWN caches (tests)."""
    global _CSV_KN, _CSV_TE, _IWN_BY_LANG
    _CSV_KN = None
    _CSV_TE = None
    _IWN_BY_LANG = {}


def _clean_gloss(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _load_csv_indexes() -> tuple[dict[str, str], dict[str, str]]:
    """Build kn_word→en_kn and te_word→en_te maps; first source wins."""
    global _CSV_KN, _CSV_TE
    if _CSV_KN is not None and _CSV_TE is not None:
        return _CSV_KN, _CSV_TE

    kn_map: dict[str, str] = {}
    te_map: dict[str, str] = {}
    for path in _CSV_SOURCES:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            kn_word = str(row.get("kn_word", "") or "").strip()
            te_word = str(row.get("te_word", "") or "").strip()
            en_kn = _clean_gloss(row.get("en_kn"))
            en_te = _clean_gloss(row.get("en_te"))
            if kn_word and en_kn and kn_word not in kn_map:
                kn_map[kn_word] = en_kn
            if te_word and en_te and te_word not in te_map:
                te_map[te_word] = en_te

    _CSV_KN, _CSV_TE = kn_map, te_map
    return kn_map, te_map


def _iwn_for(lang: str) -> Any | None:
    """Lazy IndoWordNet language object via cognate.iwn (local data)."""
    if lang in _IWN_BY_LANG:
        return _IWN_BY_LANG[lang]
    try:
        from cognate.iwn import language_enum, load_lang
    except Exception:
        _IWN_BY_LANG[lang] = None
        return None
    try:
        Language = language_enum()
        if lang == "kn":
            iwn = load_lang(Language.KANNADA)
        elif lang == "te":
            iwn = load_lang(Language.TELUGU)
        else:
            iwn = None
    except Exception:
        iwn = None
    _IWN_BY_LANG[lang] = iwn
    return iwn


def _iwn_gloss(word: str, lang: str) -> str | None:
    """First non-empty synset gloss for ``word`` via iwn.gloss / iwn.load_lang."""
    from cognate.iwn import gloss as synset_gloss

    iwn = _iwn_for(lang)
    if iwn is None:
        return None
    try:
        synsets = iwn.synsets(word)
    except Exception:
        return None
    if not synsets:
        return None
    for syn in synsets:
        text = _clean_gloss(synset_gloss(syn))
        if text:
            return text
    return None


def find_gloss(word: str, lang: str) -> str | None:
    """
    Tiered gloss lookup for a Kannada (``lang="kn"``) or Telugu (``lang="te"``) word.

    1. Exact match in gold / stream_a_glossed / stream_b_glossed (first hit).
    2. IndoWordNet synset gloss via ``cognate.iwn`` (local).
    """
    word = (word or "").strip()
    if not word:
        return None
    if lang not in {"kn", "te"}:
        raise ValueError(f"lang must be 'kn' or 'te', got {lang!r}")

    kn_map, te_map = _load_csv_indexes()
    csv_hit = kn_map.get(word) if lang == "kn" else te_map.get(word)
    if csv_hit:
        return csv_hit
    return _iwn_gloss(word, lang)


def get_gloss_or_prompt(word: str, lang: str) -> tuple[str | None, bool]:
    """
    ``found_automatically`` is True when a non-empty gloss came from CSV or IWN.
    """
    gloss = find_gloss(word, lang)
    if gloss:
        return gloss, True
    return None, False
