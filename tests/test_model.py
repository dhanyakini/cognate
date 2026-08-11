"""Tests for cognate.model train/predict round-trip."""

from __future__ import annotations

import pandas as pd
import pytest

from cognate.model import predict, save_model, load_model, train, warn_if_small


def _toy_df(n_per: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n_per):
        rows.append(
            {
                "pair_id": f"c{i}",
                "orth_sim": 0.9,
                "phon_sim": 0.85,
                "sem_sim": 0.9,
                "label": "cognate",
            }
        )
        rows.append(
            {
                "pair_id": f"f{i}",
                "orth_sim": 0.85,
                "phon_sim": 0.8,
                "sem_sim": 0.2,
                "label": "false_friend",
            }
        )
        rows.append(
            {
                "pair_id": f"u{i}",
                "orth_sim": 0.1,
                "phon_sim": 0.15,
                "sem_sim": 0.2,
                "label": "unrelated",
            }
        )
    return pd.DataFrame(rows)


def test_train_predict_roundtrip(tmp_path) -> None:
    df = _toy_df(15)
    cols = ["orth_sim", "phon_sim", "sem_sim"]
    model = train(df, cols)
    preds = predict(model, df, cols)
    assert len(preds) == len(df)
    assert set(preds) <= {"cognate", "false_friend", "unrelated"}

    path = tmp_path / "m.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert predict(loaded, df, cols) == preds


def test_small_n_warning(capsys) -> None:
    df = _toy_df(5)  # 15 rows < 100
    warn_if_small(df)
    err = capsys.readouterr().err
    assert "SMOKE TEST ONLY" in err
