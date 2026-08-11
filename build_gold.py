#!/usr/bin/env python
"""
build_gold.py — assemble the final cognate/false-friend gold CSV (v2).

Concatenates an adjudicated pilot file with a fully adjudicated 300-row
batch file. Validation failures are loud and itemized; nothing is written
on abort.

Usage:
    python build_gold.py \\
      --pilot-gold pilot_adjudication.csv \\
      --batch-gold data/batch_adjudication_full.csv \\
      --out data/gold.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

OUTPUT_COLUMNS = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "en_kn",
    "en_te",
    "gloss",
    "candidate_source",
    "final_label",
    "final_origin",
    "source_batch",
]

CONTENT_COLUMNS = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "en_kn",
    "en_te",
    "gloss",
    "candidate_source",
]


class ValidationError(Exception):
    """Data-integrity failure; main() exits 1 without writing."""


def load_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [{k: (v or "") for k, v in row.items()} for row in csv.DictReader(f)]


def blank_final_label_ids(rows: list[dict[str, str]]) -> list[str]:
    return [
        (row.get("pair_id") or "").strip() or "<missing pair_id>"
        for row in rows
        if not (row.get("final_label") or "").strip()
    ]


UNSPECIFIED_ORIGIN = "unspecified"


def require_all_final_labels(rows: list[dict[str, str]], source_name: str) -> None:
    blank = blank_final_label_ids(rows)
    if blank:
        raise ValidationError(
            f"{source_name}: {len(blank)} row(s) with blank final_label — aborting.\n"
            f"  pair_ids: {', '.join(blank)}"
        )


def normalize_origin(value: str) -> str:
    """Blank origin → 'unspecified' (deliberate scope cut, not a silent drop)."""
    stripped = (value or "").strip()
    return stripped if stripped else UNSPECIFIED_ORIGIN


def tag_rows(
    rows: list[dict[str, str]],
    *,
    source_batch: str,
) -> list[dict[str, str]]:
    tagged: list[dict[str, str]] = []
    for row in rows:
        out = {col: row.get(col, "") for col in CONTENT_COLUMNS}
        out["final_label"] = (row.get("final_label") or "").strip()
        out["final_origin"] = normalize_origin(row.get("final_origin", ""))
        out["source_batch"] = source_batch
        tagged.append(out)
    return tagged


def assert_unique_pair_ids(rows: list[dict[str, str]]) -> None:
    counts = Counter(r["pair_id"] for r in rows)
    dups = sorted(pid for pid, n in counts.items() if n > 1)
    if dups:
        raise ValidationError(
            f"duplicate pair_id across combined gold ({len(dups)} id(s)) — aborting.\n"
            f"  pair_ids: {', '.join(dups)}"
        )


def format_summary(rows: list[dict[str, str]]) -> str:
    by_batch = Counter(r["source_batch"] for r in rows)
    by_label = Counter(r["final_label"] for r in rows)
    cognates = [r for r in rows if r["final_label"] == "cognate"]
    n_unspecified = sum(
        1 for r in cognates if r["final_origin"] == UNSPECIFIED_ORIGIN
    )
    lines = [
        "=== GOLD BUILD SUMMARY ===",
        f"total rows written: {len(rows)}",
        "",
        "by source_batch:",
        f"  pilot: {by_batch.get('pilot', 0)}",
        f"  batch: {by_batch.get('batch', 0)}",
        "",
        "by final_label:",
    ]
    for label in sorted(by_label):
        lines.append(f"  {label}: {by_label[label]}")
    lines += [
        "",
        f"final_origin: {n_unspecified} of {len(cognates)} cognate rows left as "
        f"'{UNSPECIFIED_ORIGIN}' (origin typology out of scope for this pass).",
        "=== END SUMMARY ===",
    ]
    return "\n".join(lines)


def build_gold(
    *,
    pilot_gold: str | Path,
    batch_gold: str | Path,
) -> list[dict[str, str]]:
    pilot_rows = load_rows(pilot_gold)
    batch_rows = load_rows(batch_gold)

    require_all_final_labels(pilot_rows, "pilot-gold")
    require_all_final_labels(batch_rows, "batch-gold")

    combined = tag_rows(pilot_rows, source_batch="pilot") + tag_rows(
        batch_rows, source_batch="batch"
    )
    assert_unique_pair_ids(combined)
    return combined


def write_gold(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-gold", required=True)
    ap.add_argument("--batch-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    try:
        rows = build_gold(pilot_gold=args.pilot_gold, batch_gold=args.batch_gold)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    write_gold(rows, out_path)
    print(format_summary(rows))
    print(f"wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
