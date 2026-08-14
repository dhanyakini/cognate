"""Tests that demo.py delegates split/metrics to the cognate library."""

from __future__ import annotations

from pathlib import Path

import demo
from cognate.model import DEFAULT_FEATURE_COLS, load_feature_frame, load_model, split_train_test

ROOT = Path(__file__).resolve().parents[1]


def test_demo_main_returns_zero() -> None:
    assert demo.main([]) == 0


def test_demo_macro_f1_matches_library(monkeypatch) -> None:
    recorded: dict = {}
    real_evaluate = demo.evaluate_model

    def _wrap(model, test_df, feature_cols, **kwargs):
        result = real_evaluate(model, test_df, feature_cols, **kwargs)
        recorded["metrics"] = result
        recorded["pair_ids"] = list(test_df["pair_id"])
        return result

    monkeypatch.setattr(demo, "evaluate_model", _wrap)
    assert demo.main([]) == 0

    df = load_feature_frame(ROOT / "data" / "features.csv")
    _, test_df = split_train_test(df)
    model = load_model(ROOT / "data" / "model.joblib")
    lib = real_evaluate(model, test_df, list(DEFAULT_FEATURE_COLS))
    assert recorded["metrics"]["macro_f1"] == lib["macro_f1"]
    assert recorded["pair_ids"] == list(test_df["pair_id"])
