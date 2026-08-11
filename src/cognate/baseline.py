"""
baseline.py — orthographic-threshold classifier (2-way).

This baseline structurally cannot predict ``false_friend``: it only ever
returns ``cognate`` or ``unrelated``. Confusion matrices against the 3-class
gold labels will therefore always show a zero column for false_friend.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

LABELS = ["cognate", "false_friend", "unrelated"]


def classify_by_threshold(orth_sim: float, threshold: float = 0.5) -> str:
    """
    2-way orthographic baseline.

    Returns ``cognate`` if ``orth_sim >= threshold``, else ``unrelated``.
    Never returns ``false_friend``.
    """
    return "cognate" if orth_sim >= threshold else "unrelated"


def load_features(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = LABELS,
) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred, strict=True):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1
    return matrix


def print_confusion(
    matrix: list[list[int]],
    labels: list[str] = LABELS,
    *,
    file=sys.stdout,
) -> None:
    print("confusion (rows=true, cols=predicted):", file=file)
    header = " " * 14 + "".join(f"{lab[:10]:>12}" for lab in labels)
    print(header, file=file)
    for i, lab in enumerate(labels):
        row = f"{lab:<14}" + "".join(f"{matrix[i][j]:>12}" for j in range(len(labels)))
        print(row, file=file)


def evaluate_baseline(
    rows: list[dict[str, str]],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = [r["label"] for r in rows]
    y_pred = [classify_by_threshold(float(r["orth_sim"]), threshold) for r in rows]
    n = len(rows)
    correct = sum(t == p for t, p in zip(y_true, y_pred, strict=True))
    acc = correct / n if n else float("nan")
    print(f"baseline threshold={threshold}")
    print(f"accuracy: {acc:.3f}  ({correct}/{n})")
    print(f"prediction counts: {dict(Counter(y_pred))}")
    print_confusion(confusion_matrix(y_true, y_pred))
    return {"accuracy": acc, "n": n, "correct": correct}


def run(*, in_path: str | Path, threshold: float = 0.5) -> dict[str, float | int]:
    rows = load_features(in_path)
    return evaluate_baseline(rows, threshold=threshold)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    run(in_path=args.in_path, threshold=args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
