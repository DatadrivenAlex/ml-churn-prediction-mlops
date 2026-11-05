"""Command line entrypoints for data generation, training, drift, and serving."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from churn.config import DATA_DIR, EXPERIMENTS_DIR, REPORTS_DIR, project_root, registry_dir
from churn.data.generate import SCENARIOS, generate_churn_data, write_dataset
from churn.drift.psi import build_baseline, score_drift
from churn.registry.store import ModelRegistry
from churn.tracking.tracker import get_tracker
from churn.training.train import TrainConfig, load_training_frame, run_training, write_report
from churn.utils import jsonable, utc_now


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churn",
        description="Train and serve a customer churn model with local MLOps artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Write a seeded synthetic customer CSV")
    generate.add_argument("--rows", type=int, default=8000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--scenario", choices=SCENARIOS, default="baseline")
    generate.add_argument("--output", type=Path, default=DATA_DIR / "churn.csv")

    train = sub.add_parser("train", help="Cross-validate, track, and register both models")
    train.add_argument("--data", type=Path, default=DATA_DIR / "churn.csv")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--cv-folds", type=int, default=5)
    train.add_argument("--estimators", type=int, default=120)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--max-depth", type=int, default=3)
    train.add_argument("--registry", type=Path, default=None)
    train.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    train.add_argument("--report", type=Path, default=REPORTS_DIR / "training_report.json")
    train.add_argument("--tracking", choices=["auto", "file", "mlflow"], default="auto")

    drift = sub.add_parser("drift", help="PSI of a new batch against a reference")
    drift.add_argument("--current", type=Path, required=True)
    drift.add_argument("--reference", type=Path, default=None)
    drift.add_argument("--registry", type=Path, default=None)
    drift.add_argument("--alias", default="champion")
    drift.add_argument("--output", type=Path, default=REPORTS_DIR / "drift_report.json")

    serve = sub.add_parser("serve", help="Run the FastAPI scoring app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--registry", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "generate":
            _generate(args)
        elif args.command == "train":
            _train(args)
        elif args.command == "drift":
            _drift(args)
        elif args.command == "serve":
            _serve(args)
    except (ValueError, LookupError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _generate(args: argparse.Namespace) -> None:
    frame = generate_churn_data(n_rows=args.rows, seed=args.seed, scenario=args.scenario)
    write_dataset(frame, args.output)
    rate = float(frame["churn"].mean())
    print(
        f"Wrote {len(frame)} rows to {args.output} "
        f"(scenario={args.scenario}, seed={args.seed}, churn_rate={rate:.4f})"
    )


def _train(args: argparse.Namespace) -> None:
    frame = load_training_frame(args.data)
    config = TrainConfig(
        seed=args.seed,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        gb_n_estimators=args.estimators,
        gb_learning_rate=args.learning_rate,
        gb_max_depth=args.max_depth,
    )
    root = args.registry or registry_dir()
    report = run_training(
        frame,
        config,
        ModelRegistry(root),
        get_tracker(args.tracking, args.experiments),
    )
    write_report(report, args.report)
    champion = report["champion"]
    print(f"Registered {len(report['models'])} models under {root}")
    for model in report["models"]:
        metrics = model["metrics"]
        print(
            f"  {model['version']} {model['model_name']}: "
            f"cv_roc_auc={metrics['cv_roc_auc_mean']:.4f} "
            f"test_roc_auc={metrics['test_roc_auc']:.4f} "
            f"test_f1={metrics['test_f1']:.4f} "
            f"threshold={model['threshold']:.2f}"
        )
    print(
        f"Champion {champion['version']} ({champion['model_name']}) "
        f"threshold={champion['threshold']:.2f}"
    )
    print(f"Report: {args.report}")


def _drift(args: argparse.Namespace) -> None:
    current = pd.read_csv(args.current)
    if args.reference is not None:
        baseline = build_baseline(pd.read_csv(args.reference))
        source = _portable(args.reference)
    else:
        root = args.registry or registry_dir()
        baseline = ModelRegistry(root).load_baseline(alias=args.alias)
        source = f"{_portable(root)} alias={args.alias}"
    report = score_drift(current, baseline)
    report["scored_at"] = utc_now()
    report["current"] = str(args.current)
    report["reference"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    print(
        f"PSI rows={report['rows']} max_psi={report['max_psi']:.4f} "
        f"major={report['n_major']} shifted={report['n_minor_or_worse']}"
    )
    print(report["recommendation"])
    for feature in report["features"]:
        if feature["status"] != "stable":
            print(f"  {feature['feature']}: psi={feature['psi']:.4f} ({feature['status']})")
    print(f"Report: {args.output}")


def _portable(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(project_root()))
    except ValueError:
        return str(resolved)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    if args.registry is not None:
        os.environ["CHURN_REGISTRY_DIR"] = str(args.registry)
    uvicorn.run("churn.serving.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
