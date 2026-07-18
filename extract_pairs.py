#!/usr/bin/env python
"""
extract_pairs.py — mine Kannada–Telugu candidate word pairs from IndoWordNet.

This produces **Stream A** candidates: pairs whose Kannada and Telugu words
occupy the SAME IndoWordNet synset (i.e. the same concept). These are your
cognate / unrelated candidates.

It does NOT produce false-friend candidates. Those are **Stream B**: pairs that
are orthographically similar *after transliteration* but do NOT share a synset.
Generate Stream B with `python -m cognate.ff_mine` (see annotation_guidelines.md §5).

------------------------------------------------------------------------------
IMPORTANT — verify the pyiwn API before trusting this script.
pyiwn is a small, older library. The method/attribute names below (all_synsets,
synset_id, lemma_names, gloss) follow its NLTK-style interface, but you should
confirm them against the official example notebook:
    https://github.com/cfiltnlp/pyiwn/blob/master/examples/example.ipynb
If a name differs, it's a one-line fix in cognate.iwn.

Key assumption: IndoWordNet is a *linked* wordnet built by the expansion
approach, so a given concept carries the SAME numeric synset id across
languages. We match Kannada and Telugu synsets on that shared id. Sanity-check
this on a few known pairs (e.g. a common Sanskrit-derived word) before scaling.
------------------------------------------------------------------------------

Install:
    pip install -e ".[dev]"
    # On first run pyiwn downloads its data bundle; this needs network access.

Usage:
    python extract_pairs.py --out data/candidates/stream_a.csv
    python extract_pairs.py --out data/candidates/stream_a_pilot.csv --sample 40 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cognate import CSV_HEADER, bilingual_gloss
from cognate.iwn import gloss, language_enum, lemmas, load_lang, synsets_by_id


@dataclass
class Candidate:
    synset_id: str
    kn_word: str
    te_word: str
    gloss: str  # bilingual: "kn: … || te: …"


def build_candidates() -> list[Candidate]:
    Language = language_enum()
    kn = load_lang(Language.KANNADA)
    te = load_lang(Language.TELUGU)

    kn_by_id = synsets_by_id(kn)
    te_by_id = synsets_by_id(te)

    shared_ids = sorted(set(kn_by_id) & set(te_by_id))
    print(
        f"Kannada synsets: {len(kn_by_id):,} | "
        f"Telugu synsets: {len(te_by_id):,} | "
        f"shared ids: {len(shared_ids):,}",
        file=sys.stderr,
    )

    out: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    for sid in shared_ids:
        kn_g = gloss(kn_by_id[sid][0])
        te_g = gloss(te_by_id[sid][0])
        g = bilingual_gloss(kn_g, te_g)
        kn_words = {w for syn in kn_by_id[sid] for w in lemmas(syn)}
        te_words = {w for syn in te_by_id[sid] for w in lemmas(syn)}
        for kw in kn_words:
            for tw in te_words:
                key = (sid, kw, tw)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate(sid, kw, tw, g))
    print(f"generated {len(out):,} raw Kn–Te candidate pairs", file=sys.stderr)
    return out


def write_csv(rows: list[Candidate], path: str) -> None:
    # schema matches annotation_guidelines.md §6; ISO filled by normalize.py
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for i, c in enumerate(rows):
            w.writerow(
                [
                    f"A{i:06d}",
                    c.kn_word,
                    c.te_word,
                    "",
                    "",
                    c.synset_id,
                    c.gloss,
                    "shared_synset",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    print(f"wrote {len(rows):,} rows -> {path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="if >0, randomly keep this many pairs (pilot set)",
    )
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = build_candidates()
    if args.sample and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)
        print(
            f"sampled {len(rows)} pairs for the pilot (seed={args.seed})",
            file=sys.stderr,
        )
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
