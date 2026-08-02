"""
ff_mine.py — Stream B: false-friend candidate mining.

Finds Kn–Te pairs that are orthographically similar after ISO-15919
transliteration but do NOT share any IndoWordNet synset.

Usage:
    python -m cognate.ff_mine --config config.yaml --out data/candidates/stream_b.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from cognate.normalize import (
    DEFAULT_MAX_LEN,
    clean_word,
    drop_pairs_in_other,
    drop_reason,
    pair_keys,
    write_clean_csv,
)
from cognate.similarity import first_roman_char, normalized_similarity


@dataclass(frozen=True)
class LemmaRecord:
    word: str
    iso: str
    synsets: frozenset[str]
    gloss: str


@dataclass(frozen=True)
class FFCandidate:
    kn_word: str
    te_word: str
    kn_iso: str
    te_iso: str
    gloss: str
    similarity: float


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def clean_lemma_index(
    index: dict[str, set[str]],
    glosses: dict[str, str],
    *,
    max_len: int = DEFAULT_MAX_LEN,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """
    Apply the same token filters as normalize.py, merging synsets under the
    cleaned surface form so Stream B mining runs on final vocabulary.
    """
    cleaned: dict[str, set[str]] = defaultdict(set)
    cleaned_glosses: dict[str, str] = {}
    for word, sids in index.items():
        w = clean_word(word)
        if drop_reason(w, keep_multiword=False, max_len=max_len) is not None:
            continue
        cleaned[w] |= set(sids)
        if w not in cleaned_glosses:
            # Prefer a gloss attached to the cleaned form, else the raw form.
            cleaned_glosses[w] = glosses.get(w) or glosses.get(word, "")
    return dict(cleaned), cleaned_glosses


def build_records(
    index: dict[str, set[str]],
    glosses: dict[str, str],
    script: str,
    cache: dict[str, str],
    *,
    transliterate_fn=None,
    clean_vocab: bool = True,
    max_len: int = DEFAULT_MAX_LEN,
) -> list[LemmaRecord]:
    """Build lemma records; `transliterate_fn` defaults to cached aksharamukha."""
    if transliterate_fn is None:
        from cognate.transliterate import to_iso_cached as transliterate_fn

    if clean_vocab:
        index, glosses = clean_lemma_index(index, glosses, max_len=max_len)

    records: list[LemmaRecord] = []
    for word, sids in index.items():
        iso = transliterate_fn(word, script, cache)
        if not iso:
            continue
        records.append(
            LemmaRecord(
                word=word,
                iso=iso,
                synsets=frozenset(sids),
                gloss=glosses.get(word, ""),
            )
        )
    return records


def candidates_to_rows(pairs: list[FFCandidate]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, c in enumerate(pairs):
        rows.append(
            {
                "pair_id": f"B{i:06d}",
                "kn_word": c.kn_word,
                "te_word": c.te_word,
                "kn_iso": c.kn_iso,
                "te_iso": c.te_iso,
                "synset_id": "",
                "gloss": c.gloss,
                "candidate_source": "form_similar",
                "label": "",
                "origin": "",
                "annotator": "",
                "notes": "",
            }
        )
    return rows


def load_pair_keys_csv(path: str | Path) -> set[tuple[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return pair_keys(list(csv.DictReader(f)))


def _te_blocks(
    te_records: Iterable[LemmaRecord],
) -> dict[tuple[str, int], list[LemmaRecord]]:
    blocks: dict[tuple[str, int], list[LemmaRecord]] = defaultdict(list)
    for rec in te_records:
        key = (first_roman_char(rec.iso), len(rec.iso))
        blocks[key].append(rec)
    return blocks


def synsets_disjoint(a: frozenset[str], b: frozenset[str]) -> bool:
    return a.isdisjoint(b)


def should_keep_pair(
    kn: LemmaRecord,
    te: LemmaRecord,
    similarity: float,
    threshold: float,
) -> bool:
    """Keep iff similarity meets threshold and synset sets are disjoint."""
    if similarity < threshold:
        return False
    return synsets_disjoint(kn.synsets, te.synsets)


def mine_pairs(
    kn_records: list[LemmaRecord],
    te_records: list[LemmaRecord],
    *,
    threshold: float,
    length_tolerance: int,
    max_pairs: int,
) -> list[FFCandidate]:
    """Block, score, filter, rank, and cap false-friend candidates."""
    blocks = _te_blocks(te_records)
    kept: list[FFCandidate] = []
    seen: set[tuple[str, str]] = set()

    for kn in kn_records:
        first = first_roman_char(kn.iso)
        kn_len = len(kn.iso)
        for length in range(kn_len - length_tolerance, kn_len + length_tolerance + 1):
            if length < 1:
                continue
            for te in blocks.get((first, length), ()):
                key = (kn.word, te.word)
                if key in seen:
                    continue
                sim = normalized_similarity(kn.iso, te.iso)
                if not should_keep_pair(kn, te, sim, threshold):
                    continue
                seen.add(key)
                gloss = f"kn: {kn.gloss} || te: {te.gloss}"
                kept.append(
                    FFCandidate(
                        kn_word=kn.word,
                        te_word=te.word,
                        kn_iso=kn.iso,
                        te_iso=te.iso,
                        gloss=gloss,
                        similarity=sim,
                    )
                )

    kept.sort(key=lambda c: (-c.similarity, c.kn_word, c.te_word))
    return kept[:max_pairs]


def write_csv(rows: list[dict[str, str]] | list[FFCandidate], path: str | Path) -> None:
    if rows and isinstance(rows[0], FFCandidate):
        rows = candidates_to_rows(rows)  # type: ignore[arg-type]
    write_clean_csv(rows, path)  # type: ignore[arg-type]
    print(f"wrote {len(rows):,} rows -> {path}", file=sys.stderr)


def run(
    *,
    config_path: str | Path,
    out: str | Path,
    threshold: float | None = None,
    max_pairs: int | None = None,
    exclude_stream_a: str | Path | None = None,
    clean_vocab: bool = True,
    max_len: int = DEFAULT_MAX_LEN,
) -> list[dict[str, str]]:
    # Heavy deps only needed for the live mining path (not unit-tested filters).
    from cognate.iwn import language_enum, lemma_glosses, lemma_index, load_lang
    from cognate.transliterate import (
        SCRIPT_KANNADA,
        SCRIPT_TELUGU,
        load_cache,
        save_cache,
        to_iso_cached,
    )

    cfg = load_config(config_path)
    sb = cfg.get("stream_b", {})
    thr = float(threshold if threshold is not None else sb.get("similarity_threshold", 0.75))
    cap = int(max_pairs if max_pairs is not None else sb.get("max_pairs", 3000))
    length_tol = int(sb.get("length_tolerance", 2))
    cache_path = sb.get("translit_cache", "data/cache/iso15919.json")
    max_len = int(sb.get("max_len", max_len))

    cache = load_cache(cache_path)
    Language = language_enum()

    print("loading IndoWordNet (Kannada, Telugu)…", file=sys.stderr)
    kn_iwn = load_lang(Language.KANNADA)
    te_iwn = load_lang(Language.TELUGU)

    kn_index = lemma_index(kn_iwn)
    te_index = lemma_index(te_iwn)
    kn_glosses = lemma_glosses(kn_iwn)
    te_glosses = lemma_glosses(te_iwn)
    print(
        f"Kannada lemmas: {len(kn_index):,} | Telugu lemmas: {len(te_index):,}",
        file=sys.stderr,
    )

    kn_records = build_records(
        kn_index, kn_glosses, SCRIPT_KANNADA, cache,
        clean_vocab=clean_vocab, max_len=max_len,
    )
    te_records = build_records(
        te_index, te_glosses, SCRIPT_TELUGU, cache,
        clean_vocab=clean_vocab, max_len=max_len,
    )
    save_cache(cache, cache_path)
    print(
        f"after vocab clean: Kn {len(kn_records):,} | Te {len(te_records):,}",
        file=sys.stderr,
    )
    print(f"transliteration cache -> {cache_path} ({len(cache):,} entries)", file=sys.stderr)

    # Mine more than the cap when we will exclude Stream-A overlaps.
    mine_cap = cap * 3 if exclude_stream_a else cap
    pairs = mine_pairs(
        kn_records,
        te_records,
        threshold=thr,
        length_tolerance=length_tol,
        max_pairs=mine_cap,
    )
    rows = candidates_to_rows(pairs)
    print(
        f"mined {len(rows):,} pairs (threshold={thr}, mine_cap={mine_cap})",
        file=sys.stderr,
    )

    if exclude_stream_a:
        forbidden = load_pair_keys_csv(exclude_stream_a)
        before = len(rows)
        rows = drop_pairs_in_other(rows, forbidden)
        dropped = before - len(rows)
        for i, row in enumerate(rows):
            row["pair_id"] = f"B{i:06d}"
        rows = rows[:cap]
        print(
            f"excluded {dropped:,} Stream-A overlaps; "
            f"kept {len(rows):,} (<= max_pairs={cap})",
            file=sys.stderr,
        )
        leftover = pair_keys(rows) & forbidden
        if leftover:
            raise AssertionError(
                f"clean-A ∩ clean-B still non-empty ({len(leftover)} pairs)"
            )

    write_csv(rows, out)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override stream_b.similarity_threshold")
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="override stream_b.max_pairs")
    ap.add_argument(
        "--exclude-stream-a",
        default=None,
        help="clean Stream A CSV; drop any (kn,te) that also appear there",
    )
    ap.add_argument(
        "--no-clean-vocab",
        action="store_true",
        help="skip normalize-style lemma cleaning (not recommended)",
    )
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    args = ap.parse_args(argv)
    run(
        config_path=args.config,
        out=args.out,
        threshold=args.threshold,
        max_pairs=args.max_pairs,
        exclude_stream_a=args.exclude_stream_a,
        clean_vocab=not args.no_clean_vocab,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
