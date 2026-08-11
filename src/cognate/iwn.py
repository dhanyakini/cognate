"""Shared IndoWordNet loading helpers (used by Stream A and Stream B)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _import_pyiwn():
    try:
        import pyiwn
        from pyiwn import IndoWordNet, Language
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Could not import pyiwn. Install with `pip install pyiwn` "
            "(and ensure its data bundle downloaded)."
        ) from exc
    return pyiwn, IndoWordNet, Language


def load_lang(language) -> object:
    """Hits the local pyiwn bundle (no network after install)."""
    _, IndoWordNet, _ = _import_pyiwn()
    return IndoWordNet(lang=language)


def synset_id_of(synset) -> str:
    """Return synset id as a string (pyiwn exposes `synset_id()` as a method)."""
    sid = synset.synset_id() if callable(getattr(synset, "synset_id", None)) else synset.synset_id
    return str(sid)


def synsets_by_id(iwn) -> dict[str, list]:
    """Return {synset_id: [synset, ...]} for one language."""
    by_id: dict[str, list] = defaultdict(list)
    for syn in iwn.all_synsets():
        by_id[synset_id_of(syn)].append(syn)
    return by_id


def lemmas(synset) -> list[str]:
    """Prefers lemma_names(); falls back to lemmas() for older pyiwn."""
    try:
        return [str(x) for x in synset.lemma_names()]
    except AttributeError:
        return [str(getattr(x, "name", x)) for x in synset.lemmas()]


def gloss(synset) -> str:
    """Best-effort gloss/definition for annotation context."""
    for attr in ("gloss", "definition"):
        fn = getattr(synset, attr, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass
    return ""


def lemma_index(iwn) -> dict[str, set[str]]:
    """Map each lemma to the set of synset ids it appears in."""
    index: dict[str, set[str]] = defaultdict(set)
    for syn in iwn.all_synsets():
        sid = synset_id_of(syn)
        for word in lemmas(syn):
            index[word].add(sid)
    return dict(index)


def lemma_glosses(iwn) -> dict[str, str]:
    """Pick one representative gloss per lemma (first non-empty seen)."""
    out: dict[str, str] = {}
    for syn in iwn.all_synsets():
        g = gloss(syn)
        if not g:
            continue
        for word in lemmas(syn):
            if word not in out:
                out[word] = g
    return out


def language_enum() -> Any:
    """Import pyiwn only when first needed so CLI modules can load without the data bundle."""
    _, _, Language = _import_pyiwn()
    return Language
