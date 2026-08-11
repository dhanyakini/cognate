"""Tests for build_gold.py v2 — synthetic CSVs only."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "build_gold.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("build_gold", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_gold"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_gold_mod():
    return _load_mod()


FIELDS = [
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
]


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _row(
    pair_id: str,
    final_label: str,
    *,
    final_origin: str = "",
) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "kn_word": f"kn_{pair_id}",
        "te_word": f"te_{pair_id}",
        "kn_iso": f"iso_kn_{pair_id}",
        "te_iso": f"iso_te_{pair_id}",
        "en_kn": f"en_kn_{pair_id}",
        "en_te": f"en_te_{pair_id}",
        "gloss": f"gloss_{pair_id}",
        "candidate_source": "shared_synset",
        "final_label": final_label,
        "final_origin": final_origin,
    }


def _happy_inputs(tmp_path: Path) -> dict[str, Path]:
    pilot = tmp_path / "pilot.csv"
    batch = tmp_path / "batch.csv"
    out = tmp_path / "gold.csv"
    _write(
        pilot,
        [
            _row("P1", "cognate", final_origin="sanskrit"),
            _row("P2", "false_friend"),
        ],
    )
    _write(
        batch,
        [
            _row("B1", "cognate", final_origin="inherited"),
            _row("B2", "unrelated"),
            _row("B3", "false_friend"),
            _row("B4", "cognate"),  # blank origin → unspecified
        ],
    )
    return {"pilot": pilot, "batch": batch, "out": out}


def test_happy_path(tmp_path: Path, build_gold_mod) -> None:
    paths = _happy_inputs(tmp_path)
    rows = build_gold_mod.build_gold(
        pilot_gold=paths["pilot"],
        batch_gold=paths["batch"],
    )
    assert len(rows) == 6
    by_id = {r["pair_id"]: r for r in rows}
    assert by_id["P1"]["source_batch"] == "pilot"
    assert by_id["P1"]["final_origin"] == "sanskrit"
    assert by_id["B1"]["source_batch"] == "batch"
    assert by_id["B2"]["final_label"] == "unrelated"
    assert by_id["B4"]["final_origin"] == "unspecified"

    build_gold_mod.write_gold(rows, paths["out"])
    written = list(csv.DictReader(paths["out"].open(encoding="utf-8")))
    assert list(written[0].keys()) == build_gold_mod.OUTPUT_COLUMNS
    assert len(written) == 6
    summary = build_gold_mod.format_summary(rows)
    assert "total rows written: 6" in summary
    assert "pilot: 2" in summary
    assert "batch: 4" in summary
    assert "1 of 3 cognate rows left as 'unspecified'" in summary


def test_blank_final_label_aborts(tmp_path: Path, build_gold_mod) -> None:
    paths = _happy_inputs(tmp_path)
    batch_rows = list(csv.DictReader(paths["batch"].open(encoding="utf-8")))
    batch_rows[1]["final_label"] = ""
    _write(paths["batch"], batch_rows)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pilot-gold",
            str(paths["pilot"]),
            "--batch-gold",
            str(paths["batch"]),
            "--out",
            str(paths["out"]),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "B2" in result.stderr
    assert "blank final_label" in result.stderr
    assert not paths["out"].exists()


def test_blank_cognate_origin_becomes_unspecified(
    tmp_path: Path, build_gold_mod
) -> None:
    paths = _happy_inputs(tmp_path)
    pilot_rows = list(csv.DictReader(paths["pilot"].open(encoding="utf-8")))
    pilot_rows[0]["final_origin"] = ""
    _write(paths["pilot"], pilot_rows)

    rows = build_gold_mod.build_gold(
        pilot_gold=paths["pilot"],
        batch_gold=paths["batch"],
    )
    by_id = {r["pair_id"]: r for r in rows}
    assert by_id["P1"]["final_origin"] == "unspecified"
    assert by_id["B1"]["final_origin"] == "inherited"


def test_duplicate_pair_id_aborts(tmp_path: Path, build_gold_mod) -> None:
    paths = _happy_inputs(tmp_path)
    pilot_rows = list(csv.DictReader(paths["pilot"].open(encoding="utf-8")))
    pilot_rows[0]["pair_id"] = "B1"
    _write(paths["pilot"], pilot_rows)

    with pytest.raises(build_gold_mod.ValidationError) as exc:
        build_gold_mod.build_gold(
            pilot_gold=paths["pilot"],
            batch_gold=paths["batch"],
        )
    assert "B1" in str(exc.value)
    assert "duplicate pair_id" in str(exc.value)
