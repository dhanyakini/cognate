# Cognate & False-Friend Detection for Dravidian Languages (Kannada ⇌ Telugu)

**Group 110 — Dhanya & Tejaswini**
CS6120 (Natural Language Processing) course project.

A gold-standard annotated dataset, a feature-based logistic-regression
classifier, and an interactive Streamlit learning app for Kannada–Telugu
cognates, false friends, and unrelated word pairs.

## Overview

Kannada and Telugu are both Dravidian languages with heavy, uneven Sanskrit
influence, which makes their vocabulary overlap deceptive: some shared-looking
words are genuine **cognates**, some are **false friends** (similar form,
different meaning), and some are only coincidentally similar (**unrelated**).
This project builds a labeled dataset of all three categories, trains a
classifier to tell them apart from orthographic / phonetic / semantic
features, and packages the result as a learning tool.

## Quick start (recommended)

The fastest way to see the project work is the standalone demo. It loads the
shipped gold set, features, and trained model, and reproduces the paper's
numbers, the ablation, the class-weighting comparison, and example
predictions. It needs only `pandas`, `scikit-learn`, and `joblib`.

```bash
python demo.py                  # runs against data/
python demo.py --data-dir data  # explicit data directory
```

Because `data/` already contains `gold.csv`, `features.csv`, and both trained
models, `demo.py` runs directly — nothing needs to be regenerated.

## Setup (full environment)

Requires Python 3.10+. From the project root:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# then, on either OS:
pip install -e ".[dev,ml,phonetic,semantic,app]"
```

`scikit-learn`, `joblib`, and `pandas` are the core dependencies (the `ml`
extra). The optional extras add: `phonetic` (LingPy SCA backend; a
Needleman–Wunsch fallback runs without it), `semantic`
(sentence-transformers / LaBSE — needed only to recompute features, since
`data/features.csv` already ships), `app` (Streamlit), `mt`
(machine-translation glossing), and `dev` (pytest).

The models in `data/` were trained with scikit-learn 1.9.0; install that
version if you hit an `InconsistentVersionWarning` or a `multi_class`
attribute error when loading them.

## Running the tests

```bash
pip install -e ".[dev]"
pytest                 # conftest.py puts src/ on the import path
```

69 tests across 17 files. Tests needing an optional backend (LaBSE / LingPy /
Aksharamukha) skip cleanly when those aren't installed.

## Repository structure

```
demo.py                  standalone entry point (run this first)
app.py                   Streamlit learning app
demo_utils.py            helpers used by app.py
gloss_lookup.py          tiered gloss lookup (CSV → IndoWordNet)
ui_theme.py              Streamlit theming
config.yaml              pipeline configuration (thresholds, sample sizes, paths)
pyproject.toml           package definition + optional dependencies
annotation_guidelines.md label definitions and edge-case rules
README.md                this file

src/cognate/             core library
  normalize.py, transliterate.py, ff_mine.py, iwn.py, glossing.py,
  build.py, model.py, evaluate.py, baseline.py, cli.py, similarity.py
  features/              orthographic.py, phonetic.py, semantic.py

scripts/                 make_pilot.py, make_batch.py, build_labeler.py
extract_pairs.py         Stream A mining      (repo root)
build_gold.py            merge → gold.csv     (repo root)
merge_and_kappa.py       agreement + adjudication worksheet (repo root)

tests/                   unit tests (pytest)
data/                    gold.csv, features.csv, model.joblib,
                         model_noweight.joblib, labeled data, candidates/
artifacts/               label_pairs.html, label_pairs_batch.html
```

## How the dataset was built

Running the pipeline end-to-end needs the optional backends and IndoWordNet;
the shipped `data/` already reflects the outputs of these steps, so this is
documentation, not a required step. Run with the package installed
(`pip install -e .`) or `PYTHONPATH=src` set.

### 1. Candidate mining

Stream A — shared-synset candidates from IndoWordNet (cognate/unrelated pool):

```bash
python extract_pairs.py --out data/candidates/stream_a.csv
python -m cognate.normalize --in data/candidates/stream_a.csv \
  --out data/candidates/stream_a_clean.csv --stream a
```

Stream B — form-similar candidates with disjoint synsets (false-friend pool):

```bash
python -m cognate.ff_mine --config config.yaml \
  --exclude-stream-a data/candidates/stream_a_clean.csv \
  --out data/candidates/stream_b.csv
python -m cognate.normalize --in data/candidates/stream_b.csv \
  --out data/candidates/stream_b_clean.csv --stream b
```

### 2. Sampling, annotation, and the gold set

A 40-pair stratified pilot and a 300-pair main batch are drawn (via
`scripts/make_pilot.py` and `scripts/make_batch.py`), turned into a
self-contained HTML labeler (`scripts/build_labeler.py`), and labeled
independently by both annotators.

```bash
python merge_and_kappa.py data/pilot_labeled_dhanya.csv \
  data/pilot_labeled_teja.csv --out data/pilot_adjudication.csv
python build_gold.py --pilot-gold data/pilot_adjudication.csv \
  --batch-gold data/batch_adjudication_full.csv --out data/gold.csv
```

Agreement on the 300-pair batch: **68.7% raw agreement, Cohen's κ = 0.512**
(moderate). `build_gold.py` produces `data/gold.csv` — **340 rows: 159
cognate, 54 false_friend, 127 unrelated** — and aborts loudly if any row is
missing a final label. (The `final_origin` subtype field is in the schema but
was left `unspecified`; out of scope for this timeline.)

### 3. Features and model

```bash
python -m cognate.cli featurize --in data/gold.csv --out data/features.csv
python -m cognate.cli train --in data/features.csv --model-out data/model.joblib
python -m cognate.cli evaluate --in data/features.csv --model data/model.joblib
python -m cognate.cli ablate --in data/features.csv
```

Three features per pair: orthographic similarity (normalized Levenshtein),
phonetic similarity (LingPy SCA with a Needleman–Wunsch fallback), and
semantic similarity (LaBSE gloss-embedding cosine, imputed to a neutral 0.5
with a flag when a gloss is missing). Trained artifacts: `data/model.joblib`
(`class_weight="balanced"`, the reported model) and
`data/model_noweight.joblib`.

## Results

Logistic regression, `class_weight="balanced"`, stratified 70/30 split
(`random_state=42`), on the held-out 102-row test set:

| Metric | Value |
|---|---|
| Macro-F1 | 0.75 |
| Accuracy | 0.77 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cognate | 0.93 | 0.56 | 0.70 |
| false_friend | 0.40 | 0.88 | 0.55 |
| unrelated | 1.00 | 1.00 | 1.00 |

**Key findings:**
- **Phonetic similarity added no independent signal** beyond orthographic
  similarity in the ablation (macro-F1 essentially unchanged, ~0.70), likely
  because the two are highly correlated for this language pair.
- **Semantic similarity (LaBSE) is the feature that mattered** — adding it
  lifted macro-F1 to 0.75 and was what separated cognates from false friends.
- **Class weighting is not optional**: without it, false-friend recall
  collapses to 0.00 (accuracy still looks fine at 0.84, macro-F1 falls to
  0.62), which is why macro-F1 is the headline metric under this imbalance.
- `unrelated`'s perfect score is largely a negative-sampling artifact
  (orthographic mean 0.19 vs. 0.80+ for the other classes), not evidence the
  model has cracked semantics.

## Streamlit learning app

```bash
streamlit run app.py
```

The app imports the `cognate` package, so run it with the package installed
(`pip install -e .`) or set `PYTHONPATH=src` first. Three screens:

- **Learn & Practice** — teach-then-test progression through the three
  categories, unlocking sequentially, in either language direction.
- **Check Two Words** — enter a Kannada + Telugu pair for a live model
  prediction with confidence, using a tiered gloss lookup.
- **Look Up a Word** — reverse lookup across all known relationships for a
  word, tagged dataset-verified vs. model-predicted.

## Known limitations

- Inter-annotator agreement (κ = 0.512) is moderate; the cognate/false-friend
  boundary is genuinely fuzzy, and annotator bias was corrected by discussion
  rather than a fully re-measured calibration cycle.
- The `final_origin` cognate subtype was left unpopulated.
- `unrelated`'s near-perfect classification reflects the negative-sampling
  strategy; a harder negative set would give a more honest read.
- Evaluation uses a single 70/30 split rather than cross-validation.

## Future work

Independent phonetic feature design; k-fold cross-validation; threshold tuning
for the false-friend precision/recall tradeoff; harder negative sampling for
the unrelated class; and populating the `final_origin` subtype to compare
inherited cognates vs. shared borrowings.
