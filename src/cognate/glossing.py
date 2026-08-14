"""
glossing.py — attach faithful English glosses (en_kn, en_te) to candidate CSVs.

FAITHFUL-GLOSS RULE: en_kn / en_te MUST be a faithful translation of that word's
OWN native gloss. Never disambiguate, enrich, or fix a vague/weak source gloss.
If the native gloss is vague, the English stays vague. Weak rows get
needs_gloss=true (or are flagged for exclusion), never silently improved.

Lookup order per side:
  1. English WordNet via the IndoWordNet synset (Hindi/OMW pivot when available).
  2. Machine-translate the native gloss (swappable translator; disabled unless
     COGNATE_MT=1 is set — otherwise leave blank and mark needs_gloss=true).

Usage:
    python -m cognate.glossing \\
        --in data/candidates/stream_a_clean.csv \\
        --out data/candidates/stream_a_glossed.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# English-gloss rule (see module docstring): never invent or "improve" meaning.
TranslatorFn = Callable[[str], str | None]
WordNetFn = Callable[[str, str], str | None]  # (synset_id, native_gloss) -> en
LemmaGlossFn = Callable[[str, str], str]  # (word, lang) -> native gloss

_KN_LEMMA_GLOSSES: dict[str, str] | None = None
_TE_LEMMA_GLOSSES: dict[str, str] | None = None
_HI_SYNS_BY_ID: dict[str, list] | None = None


@dataclass
class GlossStats:
    total: int = 0
    glossed_via_wordnet: int = 0
    glossed_via_mt: int = 0
    needs_gloss: int = 0

    def log(self) -> None:
        print(
            "glossing summary: "
            f"total={self.total} "
            f"glossed_via_wordnet={self.glossed_via_wordnet} "
            f"glossed_via_mt={self.glossed_via_mt} "
            f"needs_gloss={self.needs_gloss}",
            file=sys.stderr,
        )


def split_bilingual_gloss(gloss: str) -> tuple[str, str]:
    """Parse `kn: … || te: …` into (kn_native, te_native)."""
    kn, te = "", ""
    if not gloss:
        return kn, te
    for part in gloss.split("||"):
        p = part.strip()
        if re.match(r"^kn\s*:", p, flags=re.I):
            kn = re.sub(r"^kn\s*:", "", p, flags=re.I).strip()
        elif re.match(r"^te\s*:", p, flags=re.I):
            te = re.sub(r"^te\s*:", "", p, flags=re.I).strip()
    return kn, te


def translate_to_english(text: str) -> str | None:
    """
    Default MT backend. Returns None unless COGNATE_MT=1 is set, so unconfigured
    environments never invent glosses.
    """
    if not text or not text.strip():
        return None
    if os.environ.get("COGNATE_MT", "").strip() not in {"1", "true", "TRUE", "yes"}:
        return None
    try:
        from deep_translator import GoogleTranslator  # pyright: ignore[reportMissingImports]
    except Exception:
        return None
    try:
        out = GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return None
    out = (out or "").strip()
    return out or None


def _load_lemma_glosses(lang_code: str) -> dict[str, str]:
    """Lazy IndoWordNet lemma → native gloss map for Kannada or Telugu."""
    try:
        from cognate.iwn import language_enum, lemma_glosses, load_lang
    except Exception:
        return {}
    try:
        Language = language_enum()
        lang = Language.KANNADA if lang_code == "kn" else Language.TELUGU
        return lemma_glosses(load_lang(lang))
    except Exception:
        return {}


def lemma_native_gloss(word: str, lang: str) -> str:
    """Best-effort native gloss for a single lemma (random / Stream B pairs)."""
    global _KN_LEMMA_GLOSSES, _TE_LEMMA_GLOSSES
    word = (word or "").strip()
    if not word:
        return ""
    if lang == "kn":
        if _KN_LEMMA_GLOSSES is None:
            _KN_LEMMA_GLOSSES = _load_lemma_glosses("kn")
        return (_KN_LEMMA_GLOSSES.get(word) or "").strip()
    if lang == "te":
        if _TE_LEMMA_GLOSSES is None:
            _TE_LEMMA_GLOSSES = _load_lemma_glosses("te")
        return (_TE_LEMMA_GLOSSES.get(word) or "").strip()
    return ""


def resolve_native_gloss(
    *,
    bilingual_native: str,
    word: str,
    lang: str,
    lemma_fn: LemmaGlossFn | None = None,
) -> str:
    """Use the pair gloss when present; otherwise look up the lemma in IWN."""
    native = (bilingual_native or "").strip()
    if native:
        return native
    lookup = lemma_fn or lemma_native_gloss
    return lookup(word, lang).strip()


def _hindi_synsets_by_id() -> dict[str, list]:
    global _HI_SYNS_BY_ID
    if _HI_SYNS_BY_ID is not None:
        return _HI_SYNS_BY_ID
    try:
        from cognate.iwn import language_enum, load_lang, synset_id_of
    except Exception:
        _HI_SYNS_BY_ID = {}
        return _HI_SYNS_BY_ID
    try:
        Language = language_enum()
        hi = load_lang(Language.HINDI)
    except Exception:
        _HI_SYNS_BY_ID = {}
        return _HI_SYNS_BY_ID
    by_id: dict[str, list] = {}
    try:
        for syn in hi.all_synsets():
            by_id.setdefault(synset_id_of(syn), []).append(syn)
    except Exception:
        by_id = {}
    _HI_SYNS_BY_ID = by_id
    return _HI_SYNS_BY_ID


def _omw_english_from_hindi_lemmas(lemmas: list[str]) -> str | None:
    """Try Open Multilingual WordNet (Hindi → English) via NLTK, if installed."""
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None
    for lemma in lemmas:
        try:
            synsets = wn.synsets(lemma, lang="hin")
        except Exception:
            continue
        for syn in synsets or []:
            definition = (syn.definition() or "").strip()
            if definition:
                return definition
    return None


def wordnet_english_gloss(synset_id: str, native_gloss: str = "") -> str | None:
    """
    Best-effort English gloss for an IndoWordNet synset via the Hindi pivot.

    IndoWordNet shares numeric synset ids across Indian languages. We load the
    Hindi synset for that id and ask OMW/NLTK for an English definition. Returns
    None when no link exists — never fabricates from the native gloss.
    """
    del native_gloss  # unused; kept for a stable WordNetFn signature
    sid = (synset_id or "").strip()
    if not sid:
        return None
    try:
        from cognate.iwn import lemmas
    except Exception:
        return None

    hi_lemmas: list[str] = []
    try:
        for syn in _hindi_synsets_by_id().get(sid, []):
            hi_lemmas = lemmas(syn)
            if hi_lemmas:
                break
    except Exception:
        return None
    if not hi_lemmas:
        return None
    return _omw_english_from_hindi_lemmas(hi_lemmas)


def gloss_one_side(
    *,
    native_gloss: str,
    synset_id: str,
    surface_word: str = "",
    wordnet_fn: WordNetFn,
    translate_fn: TranslatorFn,
    stats: GlossStats,
) -> str:
    """Return a faithful English gloss, or '' if unavailable."""
    native = (native_gloss or "").strip()
    en = wordnet_fn(synset_id, native)
    if en:
        stats.glossed_via_wordnet += 1
        return en.strip()
    if native:
        en = translate_fn(native)
        if en:
            stats.glossed_via_mt += 1
            return en.strip()
    # Random pairs may lack IWN glosses for inflected forms; translate the
    # surface word only when no native gloss exists anywhere.
    if not native and (surface_word or "").strip():
        en = translate_fn(surface_word.strip())
        if en:
            stats.glossed_via_mt += 1
            return en.strip()
    return ""


def gloss_row(
    row: dict[str, str],
    *,
    wordnet_fn: WordNetFn = wordnet_english_gloss,
    translate_fn: TranslatorFn = translate_to_english,
    lemma_fn: LemmaGlossFn | None = None,
    stats: GlossStats | None = None,
) -> dict[str, str]:
    """
    Add en_kn, en_te, needs_gloss to a copy of `row`.

    Stream A: native glosses come from the bilingual `gloss` field; synset_id
    is used for WordNet lookup on both sides (shared concept).
    Stream B / random: synset_id is blank; gloss each side from native text only.
    """
    stats = stats or GlossStats()
    stats.total += 1
    out = dict(row)
    kn_native, te_native = split_bilingual_gloss(out.get("gloss", ""))
    kn_native = resolve_native_gloss(
        bilingual_native=kn_native,
        word=out.get("kn_word", ""),
        lang="kn",
        lemma_fn=lemma_fn,
    )
    te_native = resolve_native_gloss(
        bilingual_native=te_native,
        word=out.get("te_word", ""),
        lang="te",
        lemma_fn=lemma_fn,
    )
    sid = (out.get("synset_id") or "").strip()

    en_kn = gloss_one_side(
        native_gloss=kn_native,
        synset_id=sid,
        surface_word=out.get("kn_word", ""),
        wordnet_fn=wordnet_fn,
        translate_fn=translate_fn,
        stats=stats,
    )
    en_te = gloss_one_side(
        native_gloss=te_native,
        synset_id=sid,
        surface_word=out.get("te_word", ""),
        wordnet_fn=wordnet_fn,
        translate_fn=translate_fn,
        stats=stats,
    )
    needs = not (en_kn and en_te)
    if needs:
        stats.needs_gloss += 1
    out["en_kn"] = en_kn
    out["en_te"] = en_te
    out["needs_gloss"] = "true" if needs else "false"
    return out


def gloss_rows(
    rows: list[dict[str, str]],
    *,
    wordnet_fn: WordNetFn = wordnet_english_gloss,
    translate_fn: TranslatorFn = translate_to_english,
    lemma_fn: LemmaGlossFn | None = None,
) -> tuple[list[dict[str, str]], GlossStats]:
    stats = GlossStats()
    out = [
        gloss_row(
            r,
            wordnet_fn=wordnet_fn,
            translate_fn=translate_fn,
            lemma_fn=lemma_fn,
            stats=stats,
        )
        for r in rows
    ]
    return out, stats


def write_glossed_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Preserve input column order, then append gloss columns if missing.
    fieldnames = list(rows[0].keys())
    for col in ("en_kn", "en_te", "needs_gloss"):
        if col not in fieldnames:
            fieldnames.append(col)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in fieldnames})


def run(
    *,
    in_path: str | Path,
    out_path: str | Path,
    wordnet_fn: WordNetFn = wordnet_english_gloss,
    translate_fn: TranslatorFn = translate_to_english,
) -> list[dict[str, str]]:
    with Path(in_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    glossed, stats = gloss_rows(
        rows, wordnet_fn=wordnet_fn, translate_fn=translate_fn
    )
    write_glossed_csv(glossed, out_path)
    stats.log()
    print(f"wrote {len(glossed):,} rows -> {out_path}", file=sys.stderr)
    return glossed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args(argv)
    run(in_path=args.in_path, out_path=args.out_path)


if __name__ == "__main__":
    main()
