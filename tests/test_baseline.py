"""Tests for cognate.baseline threshold classifier."""

from __future__ import annotations

from cognate.baseline import classify_by_threshold, evaluate_baseline


def test_threshold_boundary() -> None:
    assert classify_by_threshold(0.5, threshold=0.5) == "cognate"
    assert classify_by_threshold(0.49, threshold=0.5) == "unrelated"
    assert classify_by_threshold(0.9, threshold=0.8) == "cognate"
    assert classify_by_threshold(0.7, threshold=0.8) == "unrelated"


def test_baseline_never_predicts_false_friend() -> None:
    rows = [
        {"orth_sim": "0.9", "label": "cognate"},
        {"orth_sim": "0.1", "label": "unrelated"},
        {"orth_sim": "0.95", "label": "false_friend"},
    ]
    # High threshold still only cognate/unrelated.
    for thr in (0.0, 0.5, 1.0):
        preds = [classify_by_threshold(float(r["orth_sim"]), thr) for r in rows]
        assert "false_friend" not in preds


def test_evaluate_baseline_accuracy(capsys) -> None:
    rows = [
        {"orth_sim": "0.9", "label": "cognate"},
        {"orth_sim": "0.1", "label": "unrelated"},
        {"orth_sim": "0.9", "label": "false_friend"},  # pred cognate → wrong
    ]
    stats = evaluate_baseline(rows, threshold=0.5)
    assert stats["correct"] == 2
    assert stats["n"] == 3
    assert abs(float(stats["accuracy"]) - 2 / 3) < 1e-9
