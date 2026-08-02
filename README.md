# Kannada–Telugu Cognate and False-Friend Dataset

Tools and pilot data for mining and annotating Kannada–Telugu cognates,
false friends, and unrelated word pairs from IndoWordNet.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Candidate pipeline

Generate Stream A (shared-synset candidates):

```bash
python extract_pairs.py --out data/candidates/stream_a.csv
```

Generate Stream B (form-similar candidates with disjoint synsets). Prefer
mining on cleaned vocabulary and excluding clean Stream A overlaps:

```bash
python -m cognate.ff_mine \
  --config config.yaml \
  --exclude-stream-a data/candidates/stream_a_clean.csv \
  --out data/candidates/stream_b.csv
```

Normalize both streams (drops multiword / digits / internal punctuation /
overlong tokens, then re-transliterates):

```bash
python -m cognate.normalize \
  --in data/candidates/stream_a.csv \
  --out data/candidates/stream_a_clean.csv \
  --stream a

python -m cognate.normalize \
  --in data/candidates/stream_b.csv \
  --out data/candidates/stream_b_clean.csv \
  --stream b
```

Build the 40-pair stratified pilot:

```bash
python scripts/make_pilot.py \
  --stream-a data/candidates/stream_a_clean.csv \
  --stream-b data/candidates/stream_b_clean.csv \
  --n-a-hi 10 --n-a-lo 7 --n-b 15 --n-random 8 \
  --a-sim-threshold 0.60 --seed 7 \
  --out data/candidates/pilot.csv
```

## Pilot annotation

`data/pilot_glossed.csv` is the single source of truth for the 40 pilot pairs
and their English meanings. After editing it, regenerate the standalone tool:

```bash
python scripts/build_labeler.py
```

The `en_kn` and `en_te` values **must faithfully translate the corresponding
native gloss**. Never disambiguate, improve, or correct a vague or weak source
gloss; the English must remain equally vague. Flag weak rows for exclusion
instead of silently fixing their meaning.

1. Each annotator opens `label_pairs.html`, enters their name, and labels all
   40 pairs independently. The interface autosaves locally and deliberately
   hides the candidate stream. Use the exclusion checkbox for malformed pairs.
2. Each annotator selects **Export my labels** to download their labeled CSV.
3. Compute agreement and prepare adjudication:

   ```bash
   python merge_and_kappa.py \
     pilot_labeled_dhanya.csv \
     pilot_labeled_tejaswini.csv
   ```

   This reports percent agreement, Cohen's kappa, a confusion matrix, and
   disagreements, then writes `pilot_adjudication.csv`.
4. Adjudicate disagreements together and fill `final_label` (and
   `final_origin` for cognates). The completed adjudication file is the gold
   pilot.

## Scaling to a ~300-pair batch

1. Re-clean Stream A, then re-mine Stream B on the cleaned vocabulary and drop
   any remaining A∩B pairs (see commands at the end of this section).
2. Attach English glosses (WordNet via Hindi/OMW when available; otherwise MT
   only if `COGNATE_MT=1`, else `needs_gloss=true` — never invent):

   ```bash
   pip install -e ".[mt]"
   export COGNATE_MT=1
   python -c "import nltk; nltk.download('omw-1.4'); nltk.download('wordnet')"
   python -m cognate.glossing \
     --in data/candidates/stream_a_clean.csv \
     --out data/candidates/stream_a_glossed.csv
   python -m cognate.glossing \
     --in data/candidates/stream_b_clean.csv \
     --out data/candidates/stream_b_glossed.csv
   ```

3. Sample a fresh stratified batch that excludes the pilot:

   ```bash
   python scripts/make_batch.py \
     --stream-a data/candidates/stream_a_glossed.csv \
     --stream-b data/candidates/stream_b_glossed.csv \
     --exclude data/pilot_glossed.csv \
     --out data/batch_glossed.csv \
     --n-a-hi 90 --n-a-lo 70 --n-b 100 --n-random 40 \
     --a-sim-threshold 0.60 --seed 23
   ```

   This also writes `data/batch_overlap_ids.txt` (~20% of pair_ids). Both
   annotators label the overlap; the rest is split. Full-set kappa:

   ```bash
   python merge_and_kappa.py a.csv b.csv \
     --overlap-ids data/batch_overlap_ids.txt
   ```

4. Build the 300-pair labeler (fill any `needs_gloss=true` rows first):

   ```bash
   python scripts/build_labeler.py \
     --csv data/batch_glossed.csv \
     --out label_pairs_batch.html
   ```

See `annotation_guidelines.md` for label definitions and edge cases.
