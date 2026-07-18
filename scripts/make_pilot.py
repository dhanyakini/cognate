#!/usr/bin/env python
"""
scripts/make_pilot.py — stratified kappa/pilot annotation set.

Samples Stream A HI (form-similar cognate candidates) + Stream A LO
(shared-meaning dissimilar negatives) + Stream B + random Kn×Te negatives.

Expects *cleaned* CSVs from cognate.normalize (non-blank kn_iso/te_iso, gloss).

Usage:
    python scripts/make_pilot.py \\
        --stream-a data/candidates/stream_a_clean.csv \\
        --stream-b data/candidates/stream_b_clean.csv \\
        --n-a-hi 10 --n-a-lo 7 --n-b 15 --n-random 8 \\
        --a-sim-threshold 0.60 --seed 7 \\
        --out data/candidates/pilot.csv
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

from cognate import CSV_HEADER  # noqa: E402
from cognate.features.orthographic import normalized_similarity  # noqa: E402


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def blank_annotation_fields(row: dict[str, str]) -> dict[str, str]:
    out = {col: row.get(col, "") for col in CSV_HEADER}
    # Accept legacy gloss_en if a clean file was only partially migrated.
    if not out.get("gloss") and row.get("gloss_en"):
        out["gloss"] = row["gloss_en"]
    out["label"] = ""
    out["origin"] = ""
    out["annotator"] = ""
    out["notes"] = ""
    return out


def orth_sim(row: dict[str, str]) -> float:
    return normalized_similarity(row.get("kn_iso", ""), row.get("te_iso", ""))


def split_stream_a(
    rows: list[dict[str, str]],
    threshold: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hi: list[dict[str, str]] = []
    lo: list[dict[str, str]] = []
    for row in rows:
        if orth_sim(row) >= threshold:
            hi.append(row)
        else:
            lo.append(row)
    return hi, lo


def sample_rows(rows: list[dict[str, str]], n: int, rng: random.Random) -> list[dict[str, str]]:
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return [blank_annotation_fields(r) for r in rows]
    return [blank_annotation_fields(r) for r in rng.sample(rows, n)]


def stream_a_vocab(
    stream_a: list[dict[str, str]],
) -> tuple[list[str], list[str], dict[str, str], dict[str, str]]:
    kn_words: set[str] = set()
    te_words: set[str] = set()
    kn_iso: dict[str, str] = {}
    te_iso: dict[str, str] = {}
    for row in stream_a:
        kw = row.get("kn_word", "")
        tw = row.get("te_word", "")
        if kw:
            kn_words.add(kw)
            if row.get("kn_iso"):
                kn_iso[kw] = row["kn_iso"]
        if tw:
            te_words.add(tw)
            if row.get("te_iso"):
                te_iso[tw] = row["te_iso"]
    return sorted(kn_words), sorted(te_words), kn_iso, te_iso


def all_pair_keys(*row_lists: list[dict[str, str]]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for rows in row_lists:
        for row in rows:
            keys.add((row["kn_word"], row["te_word"]))
    return keys


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
    max_attempts = max(2000, n * 100)
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        kw = rng.choice(kn_words)
        tw = rng.choice(te_words)
        key = (kw, tw)
        if key in forbidden:
            continue
        forbidden.add(key)
        k_iso = kn_iso_map.get(kw) or to_iso(kw, SCRIPT_KANNADA)
        t_iso = te_iso_map.get(tw) or to_iso(tw, SCRIPT_TELUGU)
        if not k_iso or not t_iso:
            continue
        out.append(
            {
                "pair_id": "",
                "kn_word": kw,
                "te_word": tw,
                "kn_iso": k_iso,
                "te_iso": t_iso,
                "synset_id": "",
                "gloss": "",
                "candidate_source": "random",
                "label": "",
                "origin": "",
                "annotator": "",
                "notes": "",
            }
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


def assign_pair_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for i, row in enumerate(rows):
        row["pair_id"] = f"P{i:06d}"
    return rows


def assert_pilot_ok(
    rows: list[dict[str, str]],
    *,
    n_a_hi: int,
    a_sim_threshold: float,
) -> None:
    if any(not r.get("kn_iso") or not r.get("te_iso") for r in rows):
        raise AssertionError("pilot has blank kn_iso or te_iso")
    pair_ids = [r["pair_id"] for r in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("pilot pair_ids are not unique")
    sources = {r["candidate_source"] for r in rows}
    required = {"shared_synset", "form_similar", "random"}
    if not required.issubset(sources):
        raise AssertionError(f"pilot missing sources; have {sources}, need {required}")
    # Four strata: A-HI, A-LO, B, random
    hi_count = sum(
        1
        for r in rows
        if r["candidate_source"] == "shared_synset" and orth_sim(r) >= a_sim_threshold
    )
    lo_count = sum(
        1
        for r in rows
        if r["candidate_source"] == "shared_synset" and orth_sim(r) < a_sim_threshold
    )
    if hi_count < n_a_hi:
        raise AssertionError(
            f"expected >= {n_a_hi} HI shared_synset rows, found {hi_count}"
        )
    if lo_count < 1:
        raise AssertionError("pilot missing Stream A LO (dissimilar shared-synset) rows")
    if not any(r["candidate_source"] == "form_similar" for r in rows):
        raise AssertionError("pilot missing form_similar (Stream B) rows")
    if not any(r["candidate_source"] == "random" for r in rows):
        raise AssertionError("pilot missing random rows")


def build_pilot(
    stream_a_rows: list[dict[str, str]],
    stream_b_rows: list[dict[str, str]],
    *,
    n_a_hi: int,
    n_a_lo: int,
    n_b: int,
    n_random: int,
    a_sim_threshold: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    hi, lo = split_stream_a(stream_a_rows, a_sim_threshold)
    if len(hi) < n_a_hi:
        raise RuntimeError(
            f"Stream A HI band has only {len(hi)} rows "
            f"(need {n_a_hi} at threshold {a_sim_threshold})"
        )
    if len(lo) < n_a_lo:
        raise RuntimeError(
            f"Stream A LO band has only {len(lo)} rows "
            f"(need {n_a_lo} at threshold {a_sim_threshold})"
        )

    sampled_hi = sample_rows(hi, n_a_hi, rng)
    sampled_lo = sample_rows(lo, n_a_lo, rng)
    sampled_b = sample_rows(stream_b_rows, n_b, rng)

    kn_words, te_words, kn_iso, te_iso = stream_a_vocab(stream_a_rows)
    forbidden = all_pair_keys(stream_a_rows, stream_b_rows)
    # Also block anything already sampled (should already be in forbidden).
    forbidden |= all_pair_keys(sampled_hi, sampled_lo, sampled_b)

    random_rows = make_random_pairs(
        kn_words, te_words, kn_iso, te_iso, n_random, rng, forbidden
    )

    merged = dedup_rows(sampled_hi + sampled_lo + sampled_b + random_rows)
    rng.shuffle(merged)
    assign_pair_ids(merged)
    assert_pilot_ok(merged, n_a_hi=n_a_hi, a_sim_threshold=a_sim_threshold)
    return merged


def write_pilot(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({col: row.get(col, "") for col in CSV_HEADER})
    print(f"wrote {len(rows)} pilot rows -> {path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stream-a", required=True, help="cleaned Stream A CSV")
    ap.add_argument("--stream-b", required=True, help="cleaned Stream B CSV")
    ap.add_argument("--n-a-hi", type=int, default=10)
    ap.add_argument("--n-a-lo", type=int, default=7)
    ap.add_argument("--n-b", type=int, default=15)
    ap.add_argument("--n-random", type=int, default=8)
    ap.add_argument("--a-sim-threshold", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    stream_a = read_rows(args.stream_a)
    stream_b = read_rows(args.stream_b)
    rows = build_pilot(
        stream_a,
        stream_b,
        n_a_hi=args.n_a_hi,
        n_a_lo=args.n_a_lo,
        n_b=args.n_b,
        n_random=args.n_random,
        a_sim_threshold=args.a_sim_threshold,
        seed=args.seed,
    )
    write_pilot(rows, args.out)


if __name__ == "__main__":
    main()
