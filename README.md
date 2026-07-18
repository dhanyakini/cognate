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

Generate Stream B (form-similar candidates with disjoint synsets):

```bash
python -m cognate.ff_mine \
  --config config.yaml \
  --out data/candidates/stream_b.csv
```

Normalize both streams:

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

See `annotation_guidelines.md` for label definitions and edge cases.
