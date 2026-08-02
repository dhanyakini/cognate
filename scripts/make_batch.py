#!/usr/bin/env python
"""
scripts/make_batch.py — stratified ~300-pair gold batch on FRESH pairs.

Excludes every (kn_word, te_word) already in the pilot so full-set kappa is
measured on unseen items. Writes:
  - data/batch_glossed.csv  (B-prefixed pair_ids)
  - data/batch_overlap_ids.txt  (~20% of pair_ids for double-labeling)

Usage:
    python scripts/make_batch.py \\
        --stream-a data/candidates/stream_a_glossed.csv \\
        --stream-b data/candidates/stream_b_glossed.csv \\
        --exclude data/pilot_glossed.csv \\
        --out data/batch_glossed.csv \\
        --n-a-hi 90 --n-a-lo 70 --n-b 100 --n-random 40 \\
        --a-sim-threshold 0.60 --seed 23
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cognate.features.orthographic import normalized_similarity  # noqa: E402
from cognate.normalize import pair_keys  # noqa: E402

BATCH_COLUMNS = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "synset_id",
    "gloss",
    "candidate_source",
    "label",
    "origin",
    "annotator",
    "notes",
    "en_kn",
    "en_te",
    "needs_gloss",
]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def orth_sim(row: dict[str, str]) -> float:
    return normalized_similarity(row.get("kn_iso", ""), row.get("te_iso", ""))


def split_stream_a(
    rows: list[dict[str, str]],
    threshold: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hi, lo = [], []
    for row in rows:
        (hi if orth_sim(row) >= threshold else lo).append(row)
    return hi, lo


def blank_annotation(row: dict[str, str]) -> dict[str, str]:
    out = {col: row.get(col, "") for col in BATCH_COLUMNS}
    out["label"] = ""
    out["origin"] = ""
    out["annotator"] = ""
    out["notes"] = ""
    if not out.get("needs_gloss"):
        needs = not (out.get("en_kn", "").strip() and out.get("en_te", "").strip())
        out["needs_gloss"] = "true" if needs else "false"
    return out


def sample_rows(rows: list[dict[str, str]], n: int, rng: random.Random) -> list[dict[str, str]]:
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return [blank_annotation(r) for r in rows]
    return [blank_annotation(r) for r in rng.sample(rows, n)]


def stream_vocab(
    rows: list[dict[str, str]],
) -> tuple[list[str], list[str], dict[str, str], dict[str, str]]:
    kn_words: set[str] = set()
    te_words: set[str] = set()
    kn_iso: dict[str, str] = {}
    te_iso: dict[str, str] = {}
    for row in rows:
        kw, tw = row.get("kn_word", ""), row.get("te_word", "")
        if kw:
            kn_words.add(kw)
            if row.get("kn_iso"):
                kn_iso[kw] = row["kn_iso"]
        if tw:
            te_words.add(tw)
            if row.get("te_iso"):
                te_iso[tw] = row["te_iso"]
    return sorted(kn_words), sorted(te_words), kn_iso, te_iso


def make_random_pairs(
    kn_words: list[str],
    te_words: list[str],
    kn_iso_map: dict[str, str],
    te_iso_map: dict[str, str],
    n: int,
    rng: random.Random,
    forbidden: set[tuple[str, str]],
) -> list[dict[str, str]]:
    if n <= 0 or not kn_words or not te_words:
        return []
    from cognate.transliterate import SCRIPT_KANNADA, SCRIPT_TELUGU, to_iso

    out: list[dict[str, str]] = []
    attempts = 0
    max_attempts = max(5000, n * 200)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        kw, tw = rng.choice(kn_words), rng.choice(te_words)
        key = (kw, tw)
        if key in forbidden:
            continue
        forbidden.add(key)
        k_iso = kn_iso_map.get(kw) or to_iso(kw, SCRIPT_KANNADA)
        t_iso = te_iso_map.get(tw) or to_iso(tw, SCRIPT_TELUGU)
        if not k_iso or not t_iso:
            continue
        out.append(
            blank_annotation(
                {
                    "pair_id": "",
                    "kn_word": kw,
                    "te_word": tw,
                    "kn_iso": k_iso,
                    "te_iso": t_iso,
                    "synset_id": "",
                    "gloss": "",
                    "candidate_source": "random",
                    "en_kn": "",
                    "en_te": "",
                    "needs_gloss": "true",
                }
            )
        )
    if len(out) < n:
        raise RuntimeError(
            f"could only draw {len(out)}/{n} random negatives without leakage"
        )
    return out


def dedup_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (row["kn_word"], row["te_word"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def assign_batch_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for i, row in enumerate(rows):
        row["pair_id"] = f"B{i:06d}"
    return rows


def filter_unseen(
    rows: list[dict[str, str]],
    forbidden: set[tuple[str, str]],
) -> list[dict[str, str]]:
    return [r for r in rows if (r["kn_word"], r["te_word"]) not in forbidden]


def write_overlap_ids(
    pair_ids: list[str],
    path: str | Path,
    *,
    fraction: float,
    seed: int,
) -> list[str]:
    rng = random.Random(seed + 1)
    n = max(1, int(round(len(pair_ids) * fraction))) if pair_ids else 0
    n = min(n, len(pair_ids))
    chosen = sorted(rng.sample(pair_ids, n)) if n else []
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")
    return chosen


def assert_batch_ok(
    rows: list[dict[str, str]],
    *,
    pilot_keys: set[tuple[str, str]],
    n_a_hi: int,
    a_sim_threshold: float,
) -> None:
    if any(not r.get("kn_iso") or not r.get("te_iso") for r in rows):
        raise AssertionError("batch has blank kn_iso or te_iso")
    pair_ids = [r["pair_id"] for r in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("batch pair_ids are not unique")
    if any(not pid.startswith("B") for pid in pair_ids):
        raise AssertionError("batch pair_ids must use the B prefix")
    overlap = pair_keys(rows) & pilot_keys
    if overlap:
        raise AssertionError(f"batch overlaps pilot ({len(overlap)} pairs)")
    hi = sum(
        1
        for r in rows
        if r["candidate_source"] == "shared_synset" and orth_sim(r) >= a_sim_threshold
    )
    if hi < n_a_hi:
        raise AssertionError(f"expected >= {n_a_hi} HI rows, found {hi}")
    sources = {r["candidate_source"] for r in rows}
    required = {"shared_synset", "form_similar", "random"}
    if not required.issubset(sources):
        raise AssertionError(f"missing sources; have {sources}, need {required}")


def build_batch(
    stream_a: list[dict[str, str]],
    stream_b: list[dict[str, str]],
    *,
    exclude_rows: list[dict[str, str]],
    n_a_hi: int,
    n_a_lo: int,
    n_b: int,
    n_random: int,
    a_sim_threshold: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    forbidden = pair_keys(exclude_rows) | pair_keys(stream_a) | pair_keys(stream_b)
    # Sampling pools must themselves exclude pilot pairs.
    a_pool = filter_unseen(stream_a, pair_keys(exclude_rows))
    b_pool = filter_unseen(stream_b, pair_keys(exclude_rows) | pair_keys(a_pool))

    hi, lo = split_stream_a(a_pool, a_sim_threshold)
    if len(hi) < n_a_hi:
        raise RuntimeError(f"Stream A HI has only {len(hi)} (need {n_a_hi})")
    if len(lo) < n_a_lo:
        raise RuntimeError(f"Stream A LO has only {len(lo)} (need {n_a_lo})")
    if len(b_pool) < n_b:
        raise RuntimeError(f"Stream B pool has only {len(b_pool)} (need {n_b})")

    sampled_hi = sample_rows(hi, n_a_hi, rng)
    sampled_lo = sample_rows(lo, n_a_lo, rng)
    sampled_b = sample_rows(b_pool, n_b, rng)

    kn_words, te_words, kn_iso, te_iso = stream_vocab(a_pool)
    forbidden |= pair_keys(sampled_hi + sampled_lo + sampled_b)
    random_rows = make_random_pairs(
        kn_words, te_words, kn_iso, te_iso, n_random, rng, forbidden
    )

    merged = dedup_rows(sampled_hi + sampled_lo + sampled_b + random_rows)
    rng.shuffle(merged)
    assign_batch_ids(merged)
    assert_batch_ok(
        merged,
        pilot_keys=pair_keys(exclude_rows),
        n_a_hi=n_a_hi,
        a_sim_threshold=a_sim_threshold,
    )
    return merged


def write_batch(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BATCH_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in BATCH_COLUMNS})
    needs = sum(r.get("needs_gloss", "").lower() == "true" for r in rows)
    print(f"wrote {len(rows)} batch rows -> {path}", file=sys.stderr)
    print(f"needs_gloss=true: {needs}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream-a", required=True)
    ap.add_argument("--stream-b", required=True)
    ap.add_argument("--exclude", required=True, help="pilot_glossed.csv (or prior gold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap-out", default="data/batch_overlap_ids.txt")
    ap.add_argument("--overlap-fraction", type=float, default=0.20)
    ap.add_argument("--n-a-hi", type=int, default=90)
    ap.add_argument("--n-a-lo", type=int, default=70)
    ap.add_argument("--n-b", type=int, default=100)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--a-sim-threshold", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args(argv)

    rows = build_batch(
        read_rows(args.stream_a),
        read_rows(args.stream_b),
        exclude_rows=read_rows(args.exclude),
        n_a_hi=args.n_a_hi,
        n_a_lo=args.n_a_lo,
        n_b=args.n_b,
        n_random=args.n_random,
        a_sim_threshold=args.a_sim_threshold,
        seed=args.seed,
    )
    write_batch(rows, args.out)
    overlap = write_overlap_ids(
        [r["pair_id"] for r in rows],
        args.overlap_out,
        fraction=args.overlap_fraction,
        seed=args.seed,
    )
    print(
        f"wrote {len(overlap)} overlap ids "
        f"(~{100 * args.overlap_fraction:.0f}%) -> {args.overlap_out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
