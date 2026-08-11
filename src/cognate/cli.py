"""
cognate.cli — thin subcommands for featurize / baseline / train / evaluate / ablate.

Examples:
    python -m cognate.cli featurize --in data/gold.csv --out data/features.csv
    python -m cognate.cli baseline --in data/features.csv --threshold 0.5
    python -m cognate.cli train --in data/features.csv --model-out data/model.joblib
    python -m cognate.cli train --in data/features.csv --out data/model_noweight.joblib --class-weight none
    python -m cognate.cli evaluate --in data/features.csv --model data/model.joblib
    python -m cognate.cli ablate --in data/features.csv
"""

from __future__ import annotations

import argparse


def _cmd_featurize(args: argparse.Namespace) -> int:
    from cognate.build import run

    run(in_path=args.in_path, out_path=args.out_path)
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    from cognate.baseline import run

    run(in_path=args.in_path, threshold=args.threshold)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from cognate.model import parse_class_weight, run_train

    run_train(
        in_path=args.in_path,
        model_out=args.model_out,
        class_weight=parse_class_weight(args.class_weight),
    )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from cognate.evaluate import run_evaluate

    run_evaluate(in_path=args.in_path, model_path=args.model)
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    from cognate.evaluate import run_ablate

    run_ablate(in_path=args.in_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cognate.cli", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p_feat = sub.add_parser("featurize", help="assemble features from gold.csv")
    p_feat.add_argument("--in", dest="in_path", required=True)
    p_feat.add_argument("--out", dest="out_path", required=True)
    p_feat.set_defaults(func=_cmd_featurize)

    p_base = sub.add_parser("baseline", help="orthographic threshold baseline")
    p_base.add_argument("--in", dest="in_path", required=True)
    p_base.add_argument("--threshold", type=float, default=0.5)
    p_base.set_defaults(func=_cmd_baseline)

    p_train = sub.add_parser("train", help="train logistic regression")
    p_train.add_argument("--in", dest="in_path", required=True)
    p_train.add_argument(
        "--model-out",
        "--out",
        dest="model_out",
        default="data/model.joblib",
        help="output joblib path (alias: --out)",
    )
    p_train.add_argument(
        "--class-weight",
        default="balanced",
        help="'balanced' (default) or 'none'",
    )
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a trained model")
    p_eval.add_argument("--in", dest="in_path", required=True)
    p_eval.add_argument("--model", default=None, help="optional joblib model path")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_ablate = sub.add_parser("ablate", help="feature ablation table")
    p_ablate.add_argument("--in", dest="in_path", required=True)
    p_ablate.set_defaults(func=_cmd_ablate)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
