"""Tests for cognate.evaluate ablation (fresh models per step)."""

from __future__ import annotations

import pandas as pd

from cognate.evaluate import run_ablation
from cognate.model import train


def _toy_df(n_per: int = 25) -> pd.DataFrame:
    rows = []
    for i in range(n_per):
        rows.append(
            {
                "orth_sim": 0.9 + (i % 5) * 0.01,
                "phon_sim": 0.88,
                "sem_sim": 0.92,
                "label": "cognate",
            }
        )
        rows.append(
            {
                "orth_sim": 0.8 + (i % 5) * 0.01,
                "phon_sim": 0.75,
                "sem_sim": 0.15,
                "label": "false_friend",
            }
        )
        rows.append(
            {
                "orth_sim": 0.1 + (i % 5) * 0.01,
                "phon_sim": 0.12,
                "sem_sim": 0.2,
                "label": "unrelated",
            }
        )
    return pd.DataFrame(rows)


def test_ablation_trains_three_distinct_models() -> None:
    df = _toy_df()
    train_df = df.iloc[::2].reset_index(drop=True)
    test_df = df.iloc[1::2].reset_index(drop=True)
    results = run_ablation(train_df, test_df)
    assert len(results) == 3
    assert [r["name"] for r in results] == ["orth", "orth+phon", "orth+phon+sem"]
    # Distinct model objects (not one reused fitted instance).
    models = [r["model"] for r in results]
    assert len({id(m) for m in models}) == 3
    # Each step used its own feature subset.
    assert results[0]["feature_cols"] == ["orth_sim"]
    assert results[1]["feature_cols"] == ["orth_sim", "phon_sim"]
    assert results[2]["feature_cols"] == ["orth_sim", "phon_sim", "sem_sim"]
    # Predictions are lists of the right length.
    for r in results:
        assert len(r["y_pred"]) == len(test_df)
        assert "macro_f1" in r and "false_friend_f1" in r


def test_ablation_not_same_as_single_train() -> None:
    """Sanity: orth-only and full-feature models are separately fitted."""
    df = _toy_df()
    train_df = df.iloc[::2].reset_index(drop=True)
    test_df = df.iloc[1::2].reset_index(drop=True)
    results = run_ablation(train_df, test_df)
    orth_only = train(train_df, ["orth_sim"])
    # Coefficients live in different feature spaces → different n_features_in_
    assert results[0]["model"].n_features_in_ == 1
    assert results[2]["model"].n_features_in_ == 3
    assert orth_only.n_features_in_ == 1
