"""Tests for scripts/make_batch.py."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE_BATCH = ROOT / "scripts" / "make_batch.py"


def _load():
    spec = importlib.util.spec_from_file_location("make_batch", MAKE_BATCH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def make_batch():
    return _load()


def _row(
    kn: str,
    te: str,
    kn_iso: str,
    te_iso: str,
    source: str,
    *,
    pair_id: str = "",
    en_kn: str = "en kn",
    en_te: str = "en te",
) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "kn_word": kn,
        "te_word": te,
        "kn_iso": kn_iso,
        "te_iso": te_iso,
        "synset_id": "1" if source == "shared_synset" else "",
        "gloss": "kn: x || te: y",
        "candidate_source": source,
        "label": "",
        "origin": "",
        "annotator": "",
        "notes": "",
        "en_kn": en_kn,
        "en_te": en_te,
        "needs_gloss": "false",
    }


def test_make_batch_excludes_pilot_and_writes_overlap(tmp_path: Path, make_batch) -> None:
    stream_a = [
        _row(f"knHi{i}", f"teHi{i}", "niiru", "niiru", "shared_synset")
        for i in range(100)
    ] + [
        _row(f"knLo{i}", f"teLo{i}", "aaaa", "zzzzzzzz", "shared_synset")
        for i in range(80)
    ]
    stream_b = [
        _row(f"knB{i}", f"teB{i}", "kaage", "kaaki", "form_similar")
        for i in range(120)
    ]
    # Pilot contains one HI pair that must not appear in the batch.
    pilot = [stream_a[0], stream_b[0]]

    n_a_hi, n_a_lo, n_b, n_random = 10, 8, 12, 5
    rows = make_batch.build_batch(
        stream_a,
        stream_b,
        exclude_rows=pilot,
        n_a_hi=n_a_hi,
        n_a_lo=n_a_lo,
        n_b=n_b,
        n_random=n_random,
        a_sim_threshold=0.60,
        seed=23,
    )

    assert len(rows) == n_a_hi + n_a_lo + n_b + n_random
    assert all(r["pair_id"].startswith("B") for r in rows)
    assert len({r["pair_id"] for r in rows}) == len(rows)
    assert all(r["kn_iso"] and r["te_iso"] for r in rows)
    pilot_keys = {(r["kn_word"], r["te_word"]) for r in pilot}
    assert not any((r["kn_word"], r["te_word"]) in pilot_keys for r in rows)

    out = tmp_path / "batch.csv"
    make_batch.write_batch(rows, out)
    overlap_path = tmp_path / "overlap.txt"
    overlap = make_batch.write_overlap_ids(
        [r["pair_id"] for r in rows],
        overlap_path,
        fraction=0.20,
        seed=23,
    )
    assert abs(len(overlap) - round(0.20 * len(rows))) <= 1
    assert overlap_path.read_text(encoding="utf-8").strip().splitlines() == overlap
