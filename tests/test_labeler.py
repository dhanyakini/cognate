"""Regression tests for the standalone pilot labeler and its data source."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "candidates" / "pilot.csv"
GLOSSED = ROOT / "data" / "pilot_glossed.csv"
BUILD_SCRIPT = ROOT / "scripts" / "build_labeler.py"

SHARED_COLUMNS = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "gloss",
    "candidate_source",
]


def _load_build_labeler():
    spec = importlib.util.spec_from_file_location("build_labeler", BUILD_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_labeler():
    return _load_build_labeler()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pilot_glossed_is_complete_and_matches_pilot() -> None:
    pilot = _read(PILOT)
    glossed = _read(GLOSSED)

    assert len(pilot) == len(glossed) == 40
    assert len({row["pair_id"] for row in glossed}) == 40
    assert all(row["en_kn"].strip() and row["en_te"].strip() for row in glossed)

    for raw, translated in zip(pilot, glossed, strict=True):
        assert {column: raw[column] for column in SHARED_COLUMNS} == {
            column: translated[column] for column in SHARED_COLUMNS
        }

    p17 = next(row for row in glossed if row["pair_id"] == "P000017")
    assert p17["en_te"] == "a place taller than a hill; a mountain"


def test_generator_reads_csv_and_changes_embedded_value(
    tmp_path: Path, build_labeler
) -> None:
    rows = _read(GLOSSED)
    changed = "changed from the authoritative CSV <not HTML>"
    rows[0]["en_te"] = changed
    csv_path = tmp_path / "pilot_glossed.csv"
    html_path = tmp_path / "label_pairs.html"
    _write(csv_path, rows)

    build_labeler.build(
        csv_path=csv_path,
        template_path=ROOT / "label_pairs.template.html",
        out_path=html_path,
    )

    html = html_path.read_text(encoding="utf-8")
    embedded = build_labeler.extract_embedded_pairs(html)
    assert [row["pair_id"] for row in embedded] == [
        row["pair_id"] for row in rows
    ]
    assert embedded[0]["en_te"] == changed
    assert "\\u003cnot HTML>" in html


def test_generated_html_uses_local_storage_only() -> None:
    html = (ROOT / "label_pairs.html").read_text(encoding="utf-8")
    assert "window.storage" not in html
    assert "localStorage" in html


def test_kappa_disagreement_output_includes_english_meanings(
    tmp_path: Path,
) -> None:
    rows_a = _read(GLOSSED)
    rows_b = [dict(row) for row in rows_a]
    for rows, annotator in ((rows_a, "dhanya"), (rows_b, "tejaswini")):
        for row in rows:
            row["label"] = "cognate"
            row["origin"] = ""
            row["annotator"] = annotator
            row["notes"] = ""
            row["excluded"] = ""
    rows_b[0]["label"] = "false_friend"

    file_a = tmp_path / "a.csv"
    file_b = tmp_path / "b.csv"
    adjudication = tmp_path / "adjudication.csv"
    _write(file_a, rows_a)
    _write(file_b, rows_b)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "merge_and_kappa.py"),
            str(file_a),
            str(file_b),
            "--out",
            str(adjudication),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert rows_a[0]["en_kn"] in result.stdout
    assert rows_a[0]["en_te"] in result.stdout
    with adjudication.open(newline="", encoding="utf-8") as file:
        output = list(csv.DictReader(file))
    assert output[0]["en_kn"] == rows_a[0]["en_kn"]
    assert output[0]["en_te"] == rows_a[0]["en_te"]
