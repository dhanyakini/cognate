"""Tests for scripts/make_pilot.py (stratified)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from cognate import CSV_HEADER
from cognate.features.orthographic import normalized_similarity

ROOT = Path(__file__).resolve().parents[1]
MAKE_PILOT_PATH = ROOT / "scripts" / "make_pilot.py"


def _load_make_pilot():
    spec = importlib.util.spec_from_file_location("make_pilot", MAKE_PILOT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def make_pilot():
    return _load_make_pilot()


def _row(
    kn: str,
    te: str,
    kn_iso: str,
    te_iso: str,
    source: str,
    *,
    synset_id: str = "",
    gloss: str = "kn: x || te: y",
) -> dict[str, str]:
    return {
        "pair_id": "",
        "kn_word": kn,
        "te_word": te,
        "kn_iso": kn_iso,
        "te_iso": te_iso,
        "synset_id": synset_id,
        "gloss": gloss,
        "candidate_source": source,
        "label": "cognate",
        "origin": "sanskrit",
        "annotator": "alice",
        "notes": "prefilled",
    }


def test_make_pilot_stratified_schema_and_hi_band(tmp_path: Path, make_pilot) -> None:
    # HI: identical ISO forms (sim=1.0); LO: very different forms
    stream_a = [
        _row(f"knHi{i}", f"teHi{i}", "niiru", "niiru", "shared_synset", synset_id=str(i))
        for i in range(12)
    ] + [
        _row(
            f"knLo{i}",
            f"teLo{i}",
            "aaaa",
            "zzzzzzzz",
            "shared_synset",
            synset_id=str(100 + i),
        )
        for i in range(10)
    ]
    stream_b = [
        _row(f"knB{i}", f"teB{i}", "kaage", "kaaki", "form_similar")
        for i in range(20)
    ]

    n_a_hi, n_a_lo, n_b, n_random = 5, 3, 4, 2
    thr = 0.60
    rows = make_pilot.build_pilot(
        stream_a,
        stream_b,
        n_a_hi=n_a_hi,
        n_a_lo=n_a_lo,
        n_b=n_b,
        n_random=n_random,
        a_sim_threshold=thr,
        seed=7,
    )

    assert len(rows) == n_a_hi + n_a_lo + n_b + n_random
    assert len({r["pair_id"] for r in rows}) == len(rows)
    assert all(r["kn_iso"] and r["te_iso"] for r in rows)
    assert all(r["label"] == "" for r in rows)
    assert all(r["origin"] == "" for r in rows)

    hi = [
        r
        for r in rows
        if r["candidate_source"] == "shared_synset"
        and normalized_similarity(r["kn_iso"], r["te_iso"]) >= thr
    ]
    assert len(hi) >= n_a_hi

    sources = {r["candidate_source"] for r in rows}
    assert {"shared_synset", "form_similar", "random"} <= sources

    out = tmp_path / "pilot.csv"
    make_pilot.write_pilot(rows, out)
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_HEADER
        written = list(reader)
    assert "gloss_en" not in (reader.fieldnames or [])
    assert len(written) == len(rows)
    assert all(r["label"] == "" for r in written)
