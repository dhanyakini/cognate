# Kannada–Telugu Cognate and False-Friend Detection

A CS6120 course project: a gold-standard annotated dataset, a logistic
regression classifier, and an interactive Streamlit learning app for
Kannada–Telugu cognates, false friends, and unrelated word pairs.

## Overview

Kannada and Telugu are both Dravidian languages with heavy, uneven Sanskrit
influence, which makes vocabulary overlap deceptive: some shared-looking
words are genuine cognates, some are false friends (similar form, different
meaning), and some are coincidentally similar with no relationship at all.
This project builds a labeled dataset of all three categories, trains a
classifier to distinguish them from orthographic/phonetic/semantic features,
and packages the result as a learning tool.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,phonetic,semantic,app]"
pytest
```
![alt text](image.png)
`scikit-learn` and `joblib` are core dependencies. The extras add pytest,
LingPy (SCA phonetic backend), sentence-transformers (LaBSE), and Streamlit.
`pip install -e ".[dev]"` is enough to run unit tests that skip optional
backends.

## Repository structure

- `src/cognate/` — core library (normalize, transliterate, features, model, cli)
- `scripts/` — one-off dataset-construction scripts (make_pilot, make_batch, build_labeler)
- `data/` — candidate streams, labeled data, gold.csv, trained models
- `artifacts/` — standalone HTML labeling tools (`label_pairs.html`, `label_pairs_batch.html`)
- `app.py`, `demo_utils.py`, `ui_theme.py`, `gloss_lookup.py` — Streamlit app
- `annotation_guidelines.md` — label definitions and edge cases used during annotation

## 1. Candidate mining

Stream A — shared-synset candidates from IndoWordNet (cognate/unrelated pool):

```bash
python extract_pairs.py --out data/candidates/stream_a.csv
```

Stream B — form-similar candidates with disjoint synsets (false-friend pool):

```bash
python -m cognate.ff_mine \
  --config config.yaml \
  --exclude-stream-a data/candidates/stream_a_clean.csv \
  --out data/candidates/stream_b.csv
```

Normalize both:

```bash
python -m cognate.normalize --in data/candidates/stream_a.csv --out data/candidates/stream_a_clean.csv --stream a
python -m cognate.normalize --in data/candidates/stream_b.csv --out data/candidates/stream_b_clean.csv --stream b
```

(`ff_mine --exclude-stream-a` expects the *cleaned* Stream A file, so mine
Stream B after normalizing Stream A.)

## 2. Annotation

Pilot (40 pairs, stratified sample via `scripts/make_pilot.py`) and a 300-pair
batch were each independently double-labeled by two annotators using
`artifacts/label_pairs.html` (pilot) and `artifacts/label_pairs_batch.html`
(batch), built via `scripts/build_labeler.py`, then adjudicated by hand where
the annotators disagreed. See `annotation_guidelines.md` for label definitions.

Build the 300-pair batch (excludes every pair already in the pilot; also
writes `data/batch_overlap_ids.txt`):

```bash
python scripts/make_batch.py \
  --stream-a data/candidates/stream_a_glossed.csv \
  --stream-b data/candidates/stream_b_glossed.csv \
  --exclude data/pilot_glossed.csv \
  --out data/batch_glossed.csv \
  --n-a-hi 90 --n-a-lo 70 --n-b 100 --n-random 40 \
  --a-sim-threshold 0.60 --seed 23
```

Agreement on the 300-pair batch: 68.7% raw agreement, Cohen's κ = 0.512
("moderate"). One annotator systematically over-applied the `cognate` label
relative to the other (58% vs ~38% of labels) — attributed to differing
thresholds for domain/thematic similarity vs. strict shared-origin descent.
This was resolved through direct calibration discussion before adjudicating
all 94 disagreements, rather than through a full relabel-and-remeasure cycle.

Merge pilot + batch into the gold set:

```bash
python build_gold.py \
  --pilot-gold pilot_adjudication.csv \
  --batch-gold data/batch_adjudication_full.csv \
  --out data/gold.csv
```

This produces `data/gold.csv` (340 rows: 159 cognate, 54 false_friend,
127 unrelated) and aborts loudly if it detects unresolved annotator
disagreement leaking into the merge — this is intentional; do not silence
that check.

Note: the cognate-subtype field (`final_origin`: inherited / sanskrit /
other_borrowing) is present in the schema but was not populated in the final
run — all 340 rows are `unspecified`. Out of scope for this project's
timeline, not used in the results below.

## 3. Feature extraction and model

```bash
python -m cognate.cli featurize --in data/gold.csv --out data/features.csv
python -m cognate.cli train --in data/features.csv --out data/model.joblib
python -m cognate.cli evaluate --in data/features.csv --model data/model.joblib
python -m cognate.cli ablate --in data/features.csv
```

Trained artifacts: `data/model.joblib` (`class_weight="balanced"`, the
reported model) and `data/model_noweight.joblib` (`--class-weight none`).

Three features per pair: orthographic similarity (Levenshtein-based),
phonetic similarity (LingPy SCA, with a hand-rolled Needleman-Wunsch
fallback), and semantic similarity (LaBSE embedding cosine, mapped to
[0,1], imputed to 0.5 with a `semantic_imputed` flag when a gloss is missing).

## Results

Logistic regression, `class_weight="balanced"`, stratified 70/30 split
(random_state=42), evaluated on the held-out 102-row test set:

| Metric | Value |
|---|---|
| Macro-F1 | 0.750 |
| Accuracy | 0.775 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cognate | 0.931 | 0.562 | 0.701 |
| false_friend | 0.400 | 0.875 | 0.549 |
| unrelated | 1.000 | 1.000 | 1.000 |

**Key findings:**
- **Phonetic similarity added no independent signal** beyond orthographic
  similarity in the ablation (macro-F1 0.704 orth-only → 0.698 orth+phon) —
  likely because the two features are highly correlated for this language pair.
- **Semantic similarity (LaBSE) was the feature that mattered**: the full
  orth+phon+sem model raised macro-F1 from 0.704 (orth-only) to 0.750, and
  specifically rescued false_friend F1 from an orthographic-only baseline
  (0.526 → 0.549).
- **Class weighting is not optional**: without it, false_friend recall
  collapses to 0.000 (accuracy still looks fine at 0.843, macro-F1 drops to
  0.619) — accuracy alone is a misleading metric here given the class
  imbalance, which is why macro-F1 is the headline number.
- `unrelated`'s perfect separability is largely a sampling artifact of how
  negative pairs were mined (orthographic similarity mean 0.19 vs. 0.80+ for
  the other two classes), not evidence the model has cracked semantics.

## 4. Streamlit learning app

```bash
streamlit run app.py
```

Three screens:
- **Learn & Practice** — teach-then-test progression (cognate → false_friend
  → unrelated teaching stages, unlocking sequentially, then a mixed test
  stage). Bidirectional (Kannada→Telugu / Telugu→Kannada, toggle resets
  progress by design).
- **Check Two Words** — enter or select a word pair and get a live
  model prediction with confidence, using a tiered gloss lookup
  (gold.csv → glossed candidate streams → local IndoWordNet).
- **Look Up a Word** — reverse lookup across all known relationships for a
  given word, tagged as dataset-verified vs. model-predicted.

## Known limitations

- Inter-annotator agreement (κ=0.512) is moderate, not high; the underlying
  cognate/false-friend boundary is genuinely fuzzy for this language pair,
  and one annotator's bias was corrected through discussion rather than a
  fully re-measured calibration cycle.
- Cognate subtype (`final_origin`) was left unpopulated.
- Phonetic-similarity caching is configured (`config.yaml: phonetic.cache_path`)
  but not wired up in code; only semantic-similarity caching is active. This
  has no effect on the reported results, only on repeat-run speed.
- IndoWordNet fallback glosses (used when a word isn't in the gold set or
  glossed candidate streams) are in the native script (Kannada/Telugu), not
  English, unlike the primary gloss sources.
- `unrelated`'s near-perfect classification reflects the negative-sampling
  strategy more than semantic understanding; a harder negative set (e.g.
  orthographically similar but unrelated pairs) would likely lower this score
  and give a more honest read on the model's semantic reasoning.

## Future work

- Genuinely independent phonetic feature design (current signal likely
  redundant with orthographic similarity).
- 5-fold cross-validation for more stable metric estimates than a single
  70/30 split.
- Threshold tuning to explore the false_friend precision/recall tradeoff.
- Harder negative sampling for the unrelated class.
