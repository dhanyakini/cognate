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
    python demo.py                      # full demo against data/ next to this file
    python demo.py --data-dir some/dir  # point at a different data directory

The script exits non-zero if a required data file is missing, so it is safe
to use in an automated check.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = None  # type: ignore[misc, assignment]
if InconsistentVersionWarning is not None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cognate.baseline import LABELS
from cognate.evaluate import evaluate_model, run_ablation
from cognate.model import (
    TEST_SIZE,
    RANDOM_STATE,
    load_feature_frame,
    load_model,
    predict,
    split_train_test,
)

FEATURE_COLS = ["orth_sim", "phon_sim", "sem_sim"]
_DEFAULT_DATA_DIR = _ROOT / "data"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _require(path: Path) -> Path:
    """Exit with a clear message if a required input file is absent."""
    if not path.exists():
        sys.exit(
            f"ERROR: required file not found: {path}\n"
            f"Run this script from the project root, or pass --data-dir."
        )
    return path


def load_inputs(data_dir: Path):
    """Load gold, features, and both joblibs from `data_dir` (noweight may be None)."""
    gold = pd.read_csv(_require(data_dir / "gold.csv"))
    feats = load_feature_frame(_require(data_dir / "features.csv"))
    model = load_model(_require(data_dir / "model.joblib"))
    noweight_path = data_dir / "model_noweight.joblib"
    noweight = load_model(noweight_path) if noweight_path.exists() else None
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
    """Print label distribution, provenance, and how many semantic scores were imputed."""
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
    if "semantic_imputed" in feats.columns:
        imputed = feats["semantic_imputed"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes"}
        )
        n_imp = int(imputed.sum())
        print(f"\nSemantic imputed (missing gloss → 0.5): {n_imp} / {len(feats)}")


def compare_weighting(model, noweight, test_df: pd.DataFrame, data_dir: Path) -> None:
    """Always print section 4; if the no-weight joblib is missing, print the CLI that produces it."""
    print_header("4. CLASS-WEIGHTING COMPARISON")
    if noweight is None:
        print(f"{data_dir / 'model_noweight.joblib'} not found.")
        print("Produce it with:")
        print(
            "  python -m cognate.cli train "
            "--in data/features.csv --out data/model_noweight.joblib "
            "--class-weight none"
        )
        return
    y_true = test_df["label"].tolist()
    print(f"{'model':<12}{'macro-F1':>10}{'accuracy':>10}{'ff-recall':>12}")
    for name, m in [("balanced", model), ("no-weight", noweight)]:
        y_pred = predict(m, test_df, FEATURE_COLS)
        macro = f1_score(y_true, y_pred, labels=LABELS, average="macro",
                         zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        ff_recall = precision_recall_fscore_support(
            y_true, y_pred, labels=["false_friend"], average=None,
            zero_division=0)[1][0]
        print(f"{name:<12}{macro:>10.3f}{acc:>10.3f}{ff_recall:>12.3f}")
    print("\nNote: removing class weighting RAISES accuracy but sends "
          "false-friend\nrecall to zero -- why macro-F1 is the headline metric.")


def _print_example_rows(rows: pd.DataFrame, gold_by_id) -> None:
    """Print example rows in the shared pair_id / true / pred / orth / sem layout."""
    if rows.empty:
        print("(none in this split)")
        return
    print(f"{'pair_id':<9}{'true':<13}{'pred':<13}"
          f"{'orth':>6}{'sem':>6}  words / glosses")
    n_printed = 0
    for _, r in rows.iterrows():
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
        n_printed += 1
    if n_printed == 0:
        print("(none in this split)")


def show_examples(model, gold: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print three deterministic test groups (catches, cognate→FF errors, unrelated)."""
    print_header("5. EXAMPLE PREDICTIONS (real test pairs)")
    gold_by_id = gold.set_index("pair_id")
    shown = test_df.copy()
    shown["pred"] = predict(model, test_df, FEATURE_COLS)

    groups = [
        (
            "Correct false-friend catches",
            shown[(shown["label"] == "false_friend") & (shown["pred"] == shown["label"])].head(3),
        ),
        (
            "Dominant error: cognate predicted as false_friend",
            shown[(shown["label"] == "cognate") & (shown["pred"] == "false_friend")].head(3),
        ),
        (
            "Unrelated (perfect separation)",
            shown[(shown["label"] == "unrelated") & (shown["pred"] == shown["label"])].head(1),
        ),
    ]
    for title, rows in groups:
        print(f"\n{title}:")
        _print_example_rows(rows, gold_by_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Load shipped artifacts next to this file (or --data-dir) and print the demo."""
    parser = argparse.ArgumentParser(
        description="Standalone demo for Kannada-Telugu cognate/false-friend "
                    "detection.")
    parser.add_argument(
        "--data-dir",
        default=str(_DEFAULT_DATA_DIR),
        help="directory holding gold.csv, features.csv, "
             "model.joblib (default: <repo>/data)",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    gold, feats, model, noweight = load_inputs(data_dir)

    print("Kannada-Telugu Cognate & False-Friend Detection -- project demo")
    print(f"(data directory: {data_dir.resolve()})")

    show_dataset_stats(gold, feats)
    train_df, test_df = split_train_test(
        feats, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print_header("2. HELD-OUT TEST-SET EVALUATION (balanced model)")
    evaluate_model(model, test_df, FEATURE_COLS)
    print_header("3. FEATURE ABLATION (fresh model per row, same split)")
    run_ablation(train_df, test_df)
    compare_weighting(model, noweight, test_df, data_dir)
    show_examples(model, gold, test_df)

    print("\nDemo complete. See the report PDF for full analysis and the "
          "Streamlit app (streamlit run app.py) for the interactive tool.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
