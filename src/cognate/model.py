"""
model.py — balanced logistic regression on orth/phon/sem features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DEFAULT_FEATURE_COLS = ["orth_sim", "phon_sim", "sem_sim"]
TEST_SIZE = 0.3
RANDOM_STATE = 42
SMOKE_N = 100


def load_feature_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"{path}: expected a 'label' column")
    return df


def split_train_test(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def warn_if_small(df: pd.DataFrame) -> None:
    if len(df) < SMOKE_N:
        print(
            f"===== SMOKE TEST ONLY =====\n"
            f"feature frame has only {len(df)} rows (< {SMOKE_N}). "
            f"Results are not reportable.\n"
            f"===== SMOKE TEST ONLY =====",
            file=sys.stderr,
        )


def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    class_weight: str | dict | None = "balanced",
) -> LogisticRegression:
    """Fit logistic regression. ``class_weight``: ``"balanced"``, ``None``, or a dict."""
    warn_if_small(df)
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(df[feature_cols].to_numpy(), df["label"].to_numpy())
    return model


def parse_class_weight(value: str) -> str | None:
    """CLI helper: ``balanced`` → ``'balanced'``, ``none`` → ``None``."""
    v = (value or "balanced").strip().lower()
    if v in {"balanced", "balance"}:
        return "balanced"
    if v in {"none", "null", "off"}:
        return None
    raise ValueError(
        f"unknown class_weight={value!r}; use 'balanced' or 'none'"
    )


def predict(
    model: LogisticRegression,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> list[str]:
    preds = model.predict(df[feature_cols].to_numpy())
    return [str(p) for p in preds]


def save_model(model: Any, path: str | Path) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    import joblib

    return joblib.load(path)


def run_train(
    *,
    in_path: str | Path,
    model_out: str | Path,
    feature_cols: list[str] | None = None,
    class_weight: str | None = "balanced",
) -> LogisticRegression:
    feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
    df = load_feature_frame(in_path)
    warn_if_small(df)
    train_df, _test_df = split_train_test(df)
    model = train(train_df, feature_cols, class_weight=class_weight)
    save_model(model, model_out)
    print(
        f"trained on {len(train_df)} rows "
        f"(test held out: {len(df) - len(train_df)}); "
        f"features={feature_cols}; class_weight={class_weight!r}"
    )
    print(f"wrote model -> {model_out}")
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--model-out", "--out", dest="model_out", default="data/model.joblib")
    ap.add_argument(
        "--class-weight",
        default="balanced",
        help="'balanced' (default) or 'none'",
    )
    args = ap.parse_args(argv)
    run_train(
        in_path=args.in_path,
        model_out=args.model_out,
        class_weight=parse_class_weight(args.class_weight),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
