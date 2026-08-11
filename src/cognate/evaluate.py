"""
evaluate.py — metrics, confusion matrix, and feature ablations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

from cognate.baseline import LABELS, confusion_matrix, print_confusion
from cognate.model import (
    DEFAULT_FEATURE_COLS,
    load_feature_frame,
    load_model,
    predict,
    split_train_test,
    train,
    warn_if_small,
)

ABLATION_STEPS: list[tuple[str, list[str]]] = [
    ("orth", ["orth_sim"]),
    ("orth+phon", ["orth_sim", "phon_sim"]),
    ("orth+phon+sem", ["orth_sim", "phon_sim", "sem_sim"]),
]


def _false_friend_scores(
    y_true: list[str],
    y_pred: list[str],
) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["false_friend"],
        average=None,
        zero_division=0.0,
    )
    return float(precision[0]), float(recall[0]), float(f1[0])


def report_metrics(
    y_true: list[str],
    y_pred: list[str],
    *,
    title: str = "EVALUATION",
    file=sys.stdout,
) -> dict[str, float]:
    print(f"\n=== {title} ===", file=file)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            digits=3,
            zero_division=0.0,
        ),
        file=file,
    )
    macro = float(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0.0)
    )
    print(f"macro-F1: {macro:.3f}", file=file)
    ff_p, ff_r, ff_f1 = _false_friend_scores(y_true, y_pred)
    print("=====", file=file)
    print(
        f"false_friend  precision={ff_p:.3f}  recall={ff_r:.3f}  F1={ff_f1:.3f}",
        file=file,
    )
    print("=====", file=file)
    print_confusion(confusion_matrix(y_true, y_pred), file=file)
    return {
        "macro_f1": macro,
        "false_friend_precision": ff_p,
        "false_friend_recall": ff_r,
        "false_friend_f1": ff_f1,
    }


def evaluate_model(
    model: Any,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    title: str = "EVALUATION",
) -> dict[str, float]:
    y_true = [str(x) for x in test_df["label"].tolist()]
    y_pred = predict(model, test_df, feature_cols)
    return report_metrics(y_true, y_pred, title=title)


def run_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    steps: list[tuple[str, list[str]]] | None = None,
) -> list[dict[str, Any]]:
    """
    Retrain a fresh model per feature subset on the same split.

    Returns one result dict per step (never reuses a fitted model).
    """
    steps = steps or list(ABLATION_STEPS)
    results: list[dict[str, Any]] = []
    y_true = [str(x) for x in test_df["label"].tolist()]
    print("\n=== ABLATION ===")
    print(f"{'features':<18}{'macro-F1':>10}{'ff-F1':>10}")
    for name, cols in steps:
        model = train(train_df, cols)
        y_pred = predict(model, test_df, cols)
        macro = float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0.0)
        )
        _p, _r, ff_f1 = _false_friend_scores(y_true, y_pred)
        print(f"{name:<18}{macro:>10.3f}{ff_f1:>10.3f}")
        results.append(
            {
                "name": name,
                "feature_cols": list(cols),
                "model": model,
                "macro_f1": macro,
                "false_friend_f1": ff_f1,
                "y_pred": y_pred,
            }
        )
    return results


def run_evaluate(
    *,
    in_path: str | Path,
    model_path: str | Path | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
    df = load_feature_frame(in_path)
    warn_if_small(df)
    train_df, test_df = split_train_test(df)
    if model_path:
        model = load_model(model_path)
    else:
        model = train(train_df, feature_cols)
    return evaluate_model(model, test_df, feature_cols)


def run_ablate(*, in_path: str | Path) -> list[dict[str, Any]]:
    df = load_feature_frame(in_path)
    warn_if_small(df)
    train_df, test_df = split_train_test(df)
    return run_ablation(train_df, test_df)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--model", dest="model_path", default=None)
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args(argv)
    if args.ablate:
        run_ablate(in_path=args.in_path)
    else:
        run_evaluate(in_path=args.in_path, model_path=args.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
