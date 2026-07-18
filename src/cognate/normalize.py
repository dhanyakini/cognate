"""
normalize.py — clean a raw Stream A/B candidate CSV into the canonical schema.

Order per row: clean words → drop junk/multiword → re-transliterate →
fix bilingual gloss → then dedupe on (kn_word, te_word).

Usage:
    python -m cognate.normalize --in data/candidates/stream_a.csv \\
        --out data/candidates/stream_a_clean.csv --stream a
    python -m cognate.normalize --in data/candidates/stream_b.csv \\
        --out data/candidates/stream_b_clean.csv --stream b
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cognate import CSV_HEADER, bilingual_gloss
from cognate.transliterate import (
    SCRIPT_KANNADA,
    SCRIPT_TELUGU,
    load_cache,
    save_cache,
    to_iso_cached,
)

# Leading/trailing punctuation stripped from each side (guidelines / QA review).
EDGE_PUNCT = frozenset(".,;:!?\"'()")


@dataclass
class NormalizeStats:
    rows_in: int = 0
    dropped_multiword: int = 0
    dropped_numeric_or_len1: int = 0
    dropped_empty_after_clean: int = 0
    deduped: int = 0
    rows_out: int = 0

    def log(self) -> None:
        print(
            "normalize summary: "
            f"rows_in={self.rows_in} "
            f"dropped_multiword={self.dropped_multiword} "
            f"dropped_numeric_or_len1={self.dropped_numeric_or_len1} "
            f"dropped_empty_after_clean={self.dropped_empty_after_clean} "
            f"deduped={self.deduped} "
            f"rows_out={self.rows_out}",
            file=sys.stderr,
        )


@dataclass
class NormalizeConfig:
    keep_multiword: bool = False
    translit_cache_path: str = "data/cache/iso15919.json"
    te_gloss_by_id: dict[str, str] = field(default_factory=dict)


def clean_word(word: str) -> str:
    """Strip surrounding whitespace and edge punctuation; keep internal chars."""
    w = word.strip()
    while w and w[0] in EDGE_PUNCT:
        w = w[1:].lstrip()
    while w and w[-1] in EDGE_PUNCT:
        w = w[:-1].rstrip()
    return w.strip()


def is_multiword(word: str) -> bool:
    return "_" in word or bool(re.search(r"\s", word))


def drop_reason(word: str, *, keep_multiword: bool) -> str | None:
    """Return a drop category, or None if the word is keepable."""
    if not word:
        return "empty"
    if any(ch.isdigit() for ch in word) or len(word) <= 1:
        return "numeric_or_len1"
    if not keep_multiword and is_multiword(word):
        return "multiword"
    return None


def read_raw_gloss(row: dict[str, str]) -> str:
    """Accept either legacy `gloss_en` or canonical `gloss` on input."""
    return (row.get("gloss") or row.get("gloss_en") or "").strip()


def build_te_gloss_by_id() -> dict[str, str]:
    """Load Telugu IndoWordNet once and map synset_id → gloss."""
    from cognate.iwn import gloss, language_enum, load_lang, synset_id_of

    Language = language_enum()
    te = load_lang(Language.TELUGU)
    out: dict[str, str] = {}
    for syn in te.all_synsets():
        sid = synset_id_of(syn)
        if sid not in out:
            out[sid] = gloss(syn)
    return out


def fix_gloss(
    row: dict[str, str],
    stream: str,
    te_gloss_by_id: dict[str, str],
) -> str:
    raw = read_raw_gloss(row)
    if stream == "b":
        # Already bilingual (or close); normalize missing prefix if needed.
        if "||" in raw and raw.startswith("kn:"):
            return raw
        if "||" in raw:
            return raw
        return bilingual_gloss(raw, "")
    # stream a: input is Kannada-only (or already bilingual after re-extract)
    if raw.startswith("kn:") and "||" in raw:
        return raw
    kn_part = raw
    if kn_part.startswith("kn:"):
        kn_part = kn_part[3:].strip()
    sid = (row.get("synset_id") or "").strip()
    te_part = te_gloss_by_id.get(sid, "")
    return bilingual_gloss(kn_part, te_part)


def normalize_row(
    row: dict[str, str],
    stream: str,
    cfg: NormalizeConfig,
    cache: dict[str, str],
    stats: NormalizeStats,
) -> dict[str, str] | None:
    kn = clean_word(row.get("kn_word", ""))
    te = clean_word(row.get("te_word", ""))

    for side in (kn, te):
        reason = drop_reason(side, keep_multiword=cfg.keep_multiword)
        if reason == "empty":
            stats.dropped_empty_after_clean += 1
            return None
        if reason == "numeric_or_len1":
            stats.dropped_numeric_or_len1 += 1
            return None
        if reason == "multiword":
            stats.dropped_multiword += 1
            return None

    kn_iso = to_iso_cached(kn, SCRIPT_KANNADA, cache)
    te_iso = to_iso_cached(te, SCRIPT_TELUGU, cache)
    if not kn_iso or not te_iso:
        stats.dropped_empty_after_clean += 1
        return None

    gloss = fix_gloss(row, stream, cfg.te_gloss_by_id)
    synset_id = (row.get("synset_id") or "").strip() if stream == "a" else ""
    source = row.get("candidate_source") or (
        "shared_synset" if stream == "a" else "form_similar"
    )

    return {
        "pair_id": row.get("pair_id", ""),
        "kn_word": kn,
        "te_word": te,
        "kn_iso": kn_iso,
        "te_iso": te_iso,
        "synset_id": synset_id,
        "gloss": gloss,
        "candidate_source": source,
        "label": "",
        "origin": "",
        "annotator": "",
        "notes": "",
    }


def normalize_rows(
    rows: list[dict[str, str]],
    stream: str,
    cfg: NormalizeConfig,
) -> tuple[list[dict[str, str]], NormalizeStats]:
    stats = NormalizeStats(rows_in=len(rows))
    cache = load_cache(cfg.translit_cache_path)
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        out = normalize_row(row, stream, cfg, cache, stats)
        if out is None:
            continue
        key = (out["kn_word"], out["te_word"])
        if key in seen:
            stats.deduped += 1
            continue
        seen.add(key)
        cleaned.append(out)

    save_cache(cache, cfg.translit_cache_path)
    stats.rows_out = len(cleaned)
    return cleaned, stats


def write_clean_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows):
            out = {col: row.get(col, "") for col in CSV_HEADER}
            # Keep original pair_id prefix if present; else assign stream-local ids.
            if not out["pair_id"]:
                prefix = "A" if row.get("candidate_source") == "shared_synset" else "B"
                out["pair_id"] = f"{prefix}{i:06d}"
            w.writerow(out)


def run(
    *,
    in_path: str | Path,
    out_path: str | Path,
    stream: str,
    keep_multiword: bool = False,
    translit_cache: str = "data/cache/iso15919.json",
    te_gloss_by_id: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    if stream not in {"a", "b"}:
        raise ValueError("stream must be 'a' or 'b'")

    with Path(in_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    gloss_map = te_gloss_by_id
    if gloss_map is None and stream == "a":
        print("loading Telugu IndoWordNet for bilingual glosses…", file=sys.stderr)
        gloss_map = build_te_gloss_by_id()
    elif gloss_map is None:
        gloss_map = {}

    cfg = NormalizeConfig(
        keep_multiword=keep_multiword,
        translit_cache_path=translit_cache,
        te_gloss_by_id=gloss_map,
    )
    cleaned, stats = normalize_rows(rows, stream, cfg)
    write_clean_csv(cleaned, out_path)
    stats.log()
    print(f"wrote {stats.rows_out:,} rows -> {out_path}", file=sys.stderr)
    return cleaned


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, help="raw candidate CSV")
    ap.add_argument("--out", dest="out_path", required=True, help="clean CSV path")
    ap.add_argument("--stream", required=True, choices=("a", "b"))
    ap.add_argument(
        "--keep-multiword",
        action="store_true",
        help="keep underscore/space multi-word entries",
    )
    ap.add_argument(
        "--translit-cache",
        default="data/cache/iso15919.json",
        help="ISO-15919 cache path",
    )
    args = ap.parse_args(argv)
    run(
        in_path=args.in_path,
        out_path=args.out_path,
        stream=args.stream,
        keep_multiword=args.keep_multiword,
        translit_cache=args.translit_cache,
    )


if __name__ == "__main__":
    main()
