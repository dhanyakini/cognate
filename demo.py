#!/usr/bin/env python
"""
demo.py -- Standalone entry point for the Kannada-Telugu Cognate and
False-Friend Detection project.

WHAT THIS SCRIPT DOES
---------------------
This is a single, self-contained script a grader can run to see the whole
project work end to end, WITHOUT installing the optional heavy backends
(LaBSE / sentence-transformers, LingPy). It:

  1. Loads the released gold dataset (data/gold.csv) and the pre-computed
     feature table (data/features.csv), and prints dataset statistics.
  2. Loads the trained, class-balanced logistic-regression model
     (data/model.joblib) that ships with the submission.
  3. Reproduces the held-out test-set evaluation exactly as reported in the
     paper (stratified 70/30 split, random_state=42): per-class
     precision/recall/F1, macro-F1, accuracy, and the confusion matrix.
  4. Runs the feature ablation (orth -> orth+phon -> orth+phon+sem).
  5. Runs the class-weighting comparison (balanced vs. no-weight) that shows
     why accuracy is a misleading metric under class imbalance.
  6. Classifies a few example pairs from the test set and prints, for each,
     the true label, the model's prediction, and the feature values that
     drove it -- including one correct false-friend catch and the model's
     characteristic failure mode.

WHY IT USES PRE-COMPUTED FEATURES
---------------------------------
The semantic feature is produced by LaBSE and the phonetic feature can use
LingPy; both are large optional dependencies. Feature *values* for the gold
set are already saved in data/features.csv (this is exactly the file the
paper's numbers come from), so this demo needs only pandas + scikit-learn +
joblib and runs in seconds on any machine. To recompute features from
scratch, see the full pipeline commands in the report's Appendix.

USAGE
-----
    python demo.py                      # full demo against data/
    python demo.py --data-dir some/dir  # point at a different data directory
    python demo.py --check-pair KN TE   # (optional) featurize+classify one
                                        # pair; requires the optional backends

The script exits non-zero if a required data file is missing, so it is safe
to use in an automated check.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# The shipped models were pickled under a slightly newer scikit-learn; the
# version-mismatch warning is harmless for inference and only clutters output.
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Constants that MUST match training (cognate/model.py) so the reproduced
# --- split is identical to the one the paper reports. ------------------------
FEATURE_COLS = ["orth_sim", "phon_sim", "sem_sim"]
LABELS = ["cognate", "false_friend", "unrelated"]
TEST_SIZE = 0.30
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _require(path: Path) -> Path:
    """Exit with a clear message if a required input file is absent."""
    if not path.exists():
        sys.exit(f"ERROR: required file not found: {path}\n"
                 f"Run this script from the project root, or pass --data-dir.")
    return path


def load_inputs(data_dir: Path):
    """Load gold table, feature table, and the trained model from `data_dir`."""
    gold = pd.read_csv(_require(data_dir / "gold.csv"))
    feats = pd.read_csv(_require(data_dir / "features.csv"))
    model = joblib.load(_require(data_dir / "model.joblib"))
    # The no-weight model is optional; only used for the weighting comparison.
    noweight_path = data_dir / "model_noweight.joblib"
    noweight = joblib.load(noweight_path) if noweight_path.exists() else None
    return gold, feats, model, noweight


# ---------------------------------------------------------------------------
# Reporting sections
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    """Print a visually distinct section header."""
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def show_dataset_stats(gold: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Print label distribution and provenance of the gold set."""
    print_header("1. DATASET STATISTICS")
    print(f"Gold pairs: {len(gold)}")
    print("\nLabel distribution:")
    for label, count in gold["final_label"].value_counts().items():
        print(f"  {label:<14} {count:>4}  ({count / len(gold):.1%})")
    if "candidate_source" in gold:
        print("\nBy candidate source:")
        for src, count in gold["candidate_source"].value_counts().items():
            print(f"  {src:<16} {count:>4}")
    if "source_batch" in gold:
        print("\nBy annotation round:")
        for rnd, count in gold["source_batch"].value_counts().items():
            print(f"  {rnd:<16} {count:>4}")


def make_split(feats: pd.DataFrame):
    """Reproduce the exact stratified 70/30 split used for the paper."""
    train_df, test_df = train_test_split(
        feats,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=feats["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def evaluate_model(model, test_df: pd.DataFrame) -> None:
    """Print per-class metrics, macro-F1, accuracy, and confusion matrix."""
    print_header("2. HELD-OUT TEST-SET EVALUATION (balanced model)")
    y_true = test_df["label"].tolist()
    y_pred = [str(p) for p in model.predict(test_df[FEATURE_COLS].to_numpy())]

    print(classification_report(y_true, y_pred, labels=LABELS,
                                digits=3, zero_division=0))
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro",
                     zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"macro-F1: {macro:.3f}    accuracy: {acc:.3f}")

    print("\nConfusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    header = " " * 14 + "".join(f"{l[:12]:>13}" for l in LABELS)
    print(header)
    for label, row in zip(LABELS, cm):
        print(f"{label:<14}" + "".join(f"{v:>13}" for v in row))


def run_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Retrain a fresh model per feature subset and report macro / ff-F1."""
    print_header("3. FEATURE ABLATION (fresh model per row, same split)")
    steps = [
        ("orth", ["orth_sim"]),
        ("orth+phon", ["orth_sim", "phon_sim"]),
        ("orth+phon+sem", ["orth_sim", "phon_sim", "sem_sim"]),
    ]
    y_true = test_df["label"].tolist()
    print(f"{'features':<18}{'macro-F1':>10}{'ff-F1':>10}")
    for name, cols in steps:
        # class_weight='balanced' matches the reported model configuration.
        m = LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=RANDOM_STATE)
        m.fit(train_df[cols].to_numpy(), train_df["label"].to_numpy())
        y_pred = [str(p) for p in m.predict(test_df[cols].to_numpy())]
        macro = f1_score(y_true, y_pred, labels=LABELS, average="macro",
                         zero_division=0)
        ff = precision_recall_fscore_support(
            y_true, y_pred, labels=["false_friend"], average=None,
            zero_division=0)[2][0]
        print(f"{name:<18}{macro:>10.3f}{ff:>10.3f}")


def compare_weighting(model, noweight, test_df: pd.DataFrame) -> None:
    """Show the balanced vs. no-weight contrast (accuracy is misleading)."""
    if noweight is None:
        return
    print_header("4. CLASS-WEIGHTING COMPARISON")
    y_true = test_df["label"].tolist()
    print(f"{'model':<12}{'macro-F1':>10}{'accuracy':>10}{'ff-recall':>12}")
    for name, m in [("balanced", model), ("no-weight", noweight)]:
        y_pred = [str(p) for p in m.predict(test_df[FEATURE_COLS].to_numpy())]
        macro = f1_score(y_true, y_pred, labels=LABELS, average="macro",
                         zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        ff_recall = precision_recall_fscore_support(
            y_true, y_pred, labels=["false_friend"], average=None,
            zero_division=0)[1][0]
        print(f"{name:<12}{macro:>10.3f}{acc:>10.3f}{ff_recall:>12.3f}")
    print("\nNote: removing class weighting RAISES accuracy but sends "
          "false-friend\nrecall to zero -- why macro-F1 is the headline metric.")


def show_examples(model, feats: pd.DataFrame, gold: pd.DataFrame,
                  test_df: pd.DataFrame) -> None:
    """Print a few real test pairs with prediction and driving features."""
    print_header("5. EXAMPLE PREDICTIONS (real test pairs)")
    gold_by_id = gold.set_index("pair_id")
    # Recover the pair_ids for the test rows via their original index.
    test_ids = feats.loc[test_df.index, "pair_id"].tolist() \
        if test_df.index.max() < len(feats) else None
    # Robust path: re-split with indices retained so we can map ids.
    _, test_with_idx = train_test_split(
        feats, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=feats["label"])
    preds = [str(p) for p in model.predict(
        test_with_idx[FEATURE_COLS].to_numpy())]
    test_with_idx = test_with_idx.copy()
    test_with_idx["pred"] = preds

    # Show all false-friend test pairs (the interesting class) up to 6.
    ff_rows = test_with_idx[test_with_idx["label"] == "false_friend"].head(6)
    print(f"{'pair_id':<9}{'true':<13}{'pred':<13}"
          f"{'orth':>6}{'sem':>6}  words / glosses")
    for _, r in ff_rows.iterrows():
        pid = r["pair_id"]
        if pid not in gold_by_id.index:
            continue
        g = gold_by_id.loc[pid]
        mark = "OK " if r["label"] == r["pred"] else "MISS"
        kn_en = str(g.get("en_kn", ""))[:20]
        te_en = str(g.get("en_te", ""))[:20]
        print(f"{pid:<9}{r['label']:<13}{r['pred']:<13}"
              f"{r['orth_sim']:>6.2f}{r['sem_sim']:>6.2f}  "
              f"[{mark}] {g.get('kn_word','')}/{g.get('te_word','')} "
              f"= {kn_en!r} vs {te_en!r}")


# ---------------------------------------------------------------------------
# Optional: featurize + classify a single arbitrary pair (needs backends)
# ---------------------------------------------------------------------------
def check_single_pair(kn_word: str, te_word: str) -> int:
    """Featurize and classify one arbitrary pair. Requires optional backends."""
    print_header("SINGLE-PAIR CHECK")
    try:
        from cognate.features.orthographic import normalized_similarity
        from cognate.features.phonetic import phonetic_similarity
        from cognate.features.semantic import semantic_similarity
        from cognate.transliterate import (SCRIPT_KANNADA, SCRIPT_TELUGU,
                                            to_iso)
    except Exception as exc:  # pragma: no cover
        print(f"Optional backends not importable ({exc}).")
        print("Single-pair mode needs the full library + LaBSE/LingPy extras.")
        return 2
    kn_iso = to_iso(kn_word, SCRIPT_KANNADA)
    te_iso = to_iso(te_word, SCRIPT_TELUGU)
    orth = normalized_similarity(kn_iso, te_iso)
    phon = phonetic_similarity(kn_iso, te_iso)
    print(f"{kn_word} ({kn_iso})  vs  {te_word} ({te_iso})")
    print(f"orth_sim={orth:.3f}  phon_sim={phon:.3f}  "
          f"(semantic needs English glosses; see the app for live checks)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone demo for Kannada-Telugu cognate/false-friend "
                    "detection.")
    parser.add_argument("--data-dir", default="data",
                        help="directory holding gold.csv, features.csv, "
                             "model.joblib (default: data)")
    parser.add_argument("--check-pair", nargs=2, metavar=("KN", "TE"),
                        default=None,
                        help="featurize+classify one pair (needs backends)")
    args = parser.parse_args(argv)

    if args.check_pair:
        return check_single_pair(*args.check_pair)

    data_dir = Path(args.data_dir)
    gold, feats, model, noweight = load_inputs(data_dir)

    print("Kannada-Telugu Cognate & False-Friend Detection -- project demo")
    print(f"(data directory: {data_dir.resolve()})")

    show_dataset_stats(gold, feats)
    train_df, test_df = make_split(feats)
    evaluate_model(model, test_df)
    run_ablation(train_df, test_df)
    compare_weighting(model, noweight, test_df)
    show_examples(model, feats, gold, test_df)

    print("\nDemo complete. See the report PDF for full analysis and the "
          "Streamlit app (streamlit run app.py) for the interactive tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
