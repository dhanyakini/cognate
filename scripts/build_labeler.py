#!/usr/bin/env python
"""
Build the standalone labeling UI from the authoritative glossed pilot CSV.

English-gloss rule: `en_kn` and `en_te` MUST faithfully translate that word's
native gloss. Never disambiguate, improve, or correct a vague/weak source
gloss. Preserve its uncertainty; flag weak rows for exclusion instead.

Usage:
    python scripts/build_labeler.py
    python scripts/build_labeler.py --csv data/pilot_glossed.csv \
        --template label_pairs.template.html --out label_pairs.html
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "pilot_glossed.csv"
DEFAULT_TEMPLATE = ROOT / "label_pairs.template.html"
DEFAULT_OUT = ROOT / "label_pairs.html"
PAIRS_PLACEHOLDER = "__PAIRS_JSON__"
PAIRS_PATTERN = re.compile(r"const PAIRS = (\[.*?\]);\nconst LABELS", re.DOTALL)

EMBEDDED_COLUMNS = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "gloss",
    "candidate_source",
    "en_kn",
    "en_te",
]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def validate_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 40:
        raise ValueError(f"expected 40 pilot rows, found {len(rows)}")

    pair_ids = [row.get("pair_id", "") for row in rows]
    if any(not pair_id for pair_id in pair_ids):
        raise ValueError("every row must have a pair_id")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_ids must be unique")

    for row in rows:
        pair_id = row["pair_id"]
        for column in ("en_kn", "en_te"):
            if not row.get(column, "").strip():
                raise ValueError(f"{pair_id}: {column} must be non-empty")


def embedded_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select only fields needed by the standalone UI, preserving CSV order."""
    return [
        {column: row.get(column, "") for column in EMBEDDED_COLUMNS}
        for row in rows
    ]


def pairs_json(rows: list[dict[str, str]]) -> str:
    """Serialize safely for an inline script (`<` cannot close the script)."""
    payload = json.dumps(
        embedded_rows(rows),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return payload.replace("<", "\\u003c")


def render_labeler(rows: list[dict[str, str]], template: str) -> str:
    validate_rows(rows)
    if template.count(PAIRS_PLACEHOLDER) != 1:
        raise ValueError(
            f"template must contain exactly one {PAIRS_PLACEHOLDER!r} placeholder"
        )
    return template.replace(PAIRS_PLACEHOLDER, pairs_json(rows))


def build(
    csv_path: str | Path = DEFAULT_CSV,
    template_path: str | Path = DEFAULT_TEMPLATE,
    out_path: str | Path = DEFAULT_OUT,
) -> None:
    rows = read_rows(csv_path)
    template = Path(template_path).read_text(encoding="utf-8")
    rendered = render_labeler(rows, template)
    Path(out_path).write_text(rendered, encoding="utf-8")
    print(f"wrote {len(rows)} pairs -> {out_path}")


def extract_embedded_pairs(html: str) -> list[dict[str, Any]]:
    """Migration/test helper for reading the current embedded PAIRS array."""
    match = PAIRS_PATTERN.search(html)
    if not match:
        raise ValueError("could not find embedded PAIRS array")
    parsed = json.loads(match.group(1))
    if not isinstance(parsed, list):
        raise ValueError("embedded PAIRS value is not a list")
    return parsed


def template_from_html(html: str) -> str:
    """Replace only the current PAIRS JSON with the template placeholder."""
    match = PAIRS_PATTERN.search(html)
    if not match:
        raise ValueError("could not find embedded PAIRS array")
    start, end = match.span(1)
    return html[:start] + PAIRS_PLACEHOLDER + html[end:]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)
    build(args.csv, args.template, args.out)


if __name__ == "__main__":
    main()
