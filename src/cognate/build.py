"""
build.py — assemble orthographic / phonetic / semantic features from gold.csv.

Usage:
    python -m cognate.cli featurize --in data/gold.csv --out data/features.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from cognate.features.orthographic import normalized_similarity
from cognate.features.phonetic import phonetic_similarity
from cognate.features.semantic import load_cache, save_cache, semantic_similarity

# Midpoint of the (cos+1)/2 range — used when en_kn/en_te is missing.
# Neutral imputation so missing semantics ≠ "confidently dissimilar" (0.0).
SEMANTIC_IMPUTE_VALUE = 0.5

FEATURE_COLUMNS = [
    "pair_id",
    "orth_sim",
    "phon_sim",
    "sem_sim",
    "semantic_imputed",
    "label",
]


def load_gold(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]


def drop_uncertain(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    n_drop = 0
    for row in rows:
        label = (row.get("final_label") or "").strip()
        if label == "uncertain":
            n_drop += 1
            continue
        kept.append(row)
    if n_drop:
        print(
            f"WARNING: dropped {n_drop} row(s) with final_label='uncertain'",
            file=sys.stderr,
        )
    return kept


def featurize_row(
    row: dict[str, str],
    *,
    semantic_cache: dict[str, list[float]] | None = None,
) -> dict[str, str]:
    kn_iso = row.get("kn_iso", "")
    te_iso = row.get("te_iso", "")
    orth = normalized_similarity(kn_iso, te_iso)
    try:
        phon = phonetic_similarity(kn_iso, te_iso)
    except ValueError:
        # LingPy SCA rejects some ISO-15919 sequences; NW backend always works.
        phon = phonetic_similarity(kn_iso, te_iso, backend="nw")
    sem = semantic_similarity(
        row.get("en_kn", ""),
        row.get("en_te", ""),
        cache=semantic_cache,
    )
    imputed = sem is None
    if imputed:
        sem = SEMANTIC_IMPUTE_VALUE
    return {
        "pair_id": row.get("pair_id", ""),
        "orth_sim": f"{orth:.6f}",
        "phon_sim": f"{phon:.6f}",
        "sem_sim": f"{float(sem):.6f}",
        "semantic_imputed": "True" if imputed else "False",
        "label": (row.get("final_label") or "").strip(),
    }


def featurize(
    rows: list[dict[str, str]],
    *,
    semantic_cache: dict[str, list[float]] | None = None,
) -> list[dict[str, str]]:
    rows = drop_uncertain(rows)
    return [featurize_row(row, semantic_cache=semantic_cache) for row in rows]


def write_features(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(
    *,
    in_path: str | Path,
    out_path: str | Path,
    semantic_cache_path: str | Path | None = "data/cache/semantic_embeddings.json",
) -> list[dict[str, str]]:
    gold = load_gold(in_path)
    cache: dict[str, list[float]] = {}
    if semantic_cache_path:
        cache = load_cache(semantic_cache_path)
    features = featurize(gold, semantic_cache=cache)
    if semantic_cache_path:
        save_cache(cache, semantic_cache_path)
    write_features(features, out_path)
    n_imputed = sum(1 for r in features if r["semantic_imputed"] == "True")
    print(f"wrote {len(features)} rows -> {out_path}")
    print(f"semantic_imputed=True: {n_imputed}")
    return features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    args = ap.parse_args(argv)
    run(in_path=args.in_path, out_path=args.out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
