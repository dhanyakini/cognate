#!/usr/bin/env python
"""
merge_and_kappa.py — compare two annotators' labels and prep adjudication.

Run after both people export from the labeling tool:
    python merge_and_kappa.py pilot_labeled_dhanya.csv pilot_labeled_tejaswini.csv
    python merge_and_kappa.py a.csv b.csv --out pilot_adjudication.csv

For the full-set kappa (double-labeled overlap only):
    python merge_and_kappa.py a.csv b.csv --overlap-ids data/batch_overlap_ids.txt

It prints percent agreement, Cohen's kappa (with and without 'uncertain'),
and a confusion matrix, then writes an adjudication worksheet: both labels
side by side, an `agree` flag, and blank `final_label` / `final_origin`
columns for you to fill in together. That filled worksheet becomes your gold set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

LABELS = ["cognate", "false_friend", "unrelated", "uncertain"]


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "pair_id" not in df.columns or "label" not in df.columns:
        sys.exit(f"{path}: expected columns pair_id and label")
    return df


def load_overlap_ids(path: str | Path) -> set[str]:
    text = Path(path).read_text(encoding="utf-8")
    return {line.strip() for line in text.splitlines() if line.strip()}


def annot_name(df, fallback):
    vals = [v for v in df.get("annotator", pd.Series([], dtype=str)).unique() if v.strip()]
    return vals[0] if len(vals) == 1 else fallback


def cohen_kappa(a, b, labels):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = 0.0
    for c in labels:
        pa = sum(x == c for x in a) / n
        pb = sum(y == c for y in b) / n
        pe += pa * pb
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def confusion(a, b, labels):
    idx = {c: i for i, c in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        if x in idx and y in idx:
            m[idx[x]][idx[y]] += 1
    return m


def report(a, b, labels, title):
    n = len(a)
    if n == 0:
        print(f"\n{title}: no comparable rows.")
        return
    agree = sum(x == y for x, y in zip(a, b))
    print(f"\n{title}  (n={n})")
    print(f"  percent agreement: {100*agree/n:.1f}%  ({agree}/{n})")
    k = cohen_kappa(a, b, labels)
    print(f"  Cohen's kappa:     {k:.3f}  ({kappa_reading(k)})")
    m = confusion(a, b, labels)
    w = max(12, *(len(l) for l in labels))
    print("  confusion (rows = A, cols = B):")
    print("    " + "".join(f"{l[:10]:>12}" for l in labels))
    for i, l in enumerate(labels):
        print(f"    {l:<{w}}" + "".join(f"{m[i][j]:>12}" for j in range(len(labels))))


def kappa_reading(k):
    if k != k:
        return "n/a"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--out", default="pilot_adjudication.csv")
    ap.add_argument(
        "--overlap-ids",
        default=None,
        help="optional file of pair_ids; kappa is computed on this subset only",
    )
    args = ap.parse_args(argv)

    A, B = load(args.file_a), load(args.file_b)
    na = annot_name(A, "A")
    nb = annot_name(B, "B")
    if na == nb:
        na, nb = na + "_A", nb + "_B"
    print(f"Annotator A = {na}   Annotator B = {nb}")

    keep = ["pair_id", "kn_word", "te_word", "kn_iso", "te_iso",
            "en_kn", "en_te", "gloss", "candidate_source"]
    keep = [c for c in keep if c in A.columns]
    m = A[keep].merge(
        A[["pair_id", "label", "origin", "notes", "excluded"]],
        on="pair_id").merge(
        B[["pair_id", "label", "origin", "notes", "excluded"]],
        on="pair_id", suffixes=("_"+na, "_"+nb))

    if args.overlap_ids:
        ids = load_overlap_ids(args.overlap_ids)
        before = len(m)
        m = m[m["pair_id"].isin(ids)].copy()
        print(f"overlap filter: {len(m)}/{before} rows (ids file has {len(ids)})")

    total = len(m)
    la, lb = "label_"+na, "label_"+nb
    ea, eb = "excluded_"+na, "excluded_"+nb
    excluded = m[(m[ea].str.strip() != "") | (m[eb].str.strip() != "")]
    unlabeled = m[(m[la].str.strip() == "") | (m[lb].str.strip() == "")]
    comparable = m[(m[la].str.strip() != "") & (m[lb].str.strip() != "")
                   & (m[ea].str.strip() == "") & (m[eb].str.strip() == "")]

    print(f"\nrows merged: {total} | comparable: {len(comparable)} | "
          f"excluded by someone: {len(excluded)} | unlabeled by someone: {len(unlabeled)}")

    title = "ALL FOUR LABELS"
    if args.overlap_ids:
        title += " (OVERLAP ONLY)"
    report(list(comparable[la]), list(comparable[lb]), LABELS, title)
    no_unc = comparable[(comparable[la] != "uncertain") & (comparable[lb] != "uncertain")]
    report(list(no_unc[la]), list(no_unc[lb]),
           ["cognate", "false_friend", "unrelated"], "EXCLUDING 'uncertain'")

    # adjudication worksheet
    m["agree"] = (m[la] == m[lb]).map({True: "yes", False: "NO"})
    m["final_label"] = m.apply(lambda r: r[la] if r[la] == r[lb] else "", axis=1)
    m["final_origin"] = ""
    m.to_csv(args.out, index=False)
    dis = m[m["agree"] == "NO"]
    print(f"\ndisagreements to adjudicate: {len(dis)}")
    for _, r in dis.iterrows():
        print(f"  {r['pair_id']}  {r.get('kn_word','')}/{r.get('te_word','')}")
        print(f"      KN: {r.get('en_kn','')}  |  TE: {r.get('en_te','')}")
        print(f"      [{na}={r[la]} | {nb}={r[lb]}]")
    print(f"\nwrote adjudication worksheet -> {args.out}")
    print("Fill in `final_label` (and `final_origin` for cognates) together; that file is your gold set.")


if __name__ == "__main__":
    main()
