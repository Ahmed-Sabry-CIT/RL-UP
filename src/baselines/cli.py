"""Command-line interface: ``python -m baselines <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import datasets as ds
from .config import load_config
from .models import available_models
from .pipeline import (
    build_report,
    load_prepared,
    prepare_dataset,
    run_baseline,
    verify_artifacts,
)

LOGGER = logging.getLogger("baselines")


def _setup_logging(log_dir: Path, verbose: bool) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_dir / "baselines.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("lightgbm", "xgboost", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _datasets(argument: str) -> list[str]:
    if argument in ("all", None):
        return list(ds.DATASETS)
    selected = [item.strip() for item in argument.split(",") if item.strip()]
    unknown = [item for item in selected if item not in ds.DATASETS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Choose from {list(ds.DATASETS)}")
    return selected


def _models(argument: str) -> list[str]:
    installed = available_models()
    if argument in ("all", None):
        return installed
    selected = [item.strip() for item in argument.split(",") if item.strip()]
    missing = [item for item in selected if item not in installed]
    if missing:
        raise SystemExit(
            f"Model backend(s) not installed in this interpreter: {missing}. "
            "Install with `pip install -r requirements.txt`."
        )
    return selected


def _prepare(config, names, allow_synthetic: bool) -> None:
    for name in names:
        prepare_dataset(config, name, allow_synthetic=allow_synthetic)


def cmd_prepare(args, config) -> int:
    _prepare(config, _datasets(args.dataset), allow_synthetic=not args.strict)
    return 0


def cmd_run(args, config) -> int:
    names = _datasets(args.dataset)
    models = _models(args.model)
    if not models:
        raise SystemExit("No model backend available; install lightgbm and/or xgboost")

    prepared_any = False
    for dataset in names:
        try:
            prepared = load_prepared(config, dataset)
            prepared_any = True
        except FileNotFoundError:
            LOGGER.info("%s: no prepared artefacts, preparing now", dataset)
            prepared = prepare_dataset(config, dataset, allow_synthetic=not args.strict)
            prepared_any = True

        if prepared.schema.get("source") == "synthetic":
            LOGGER.warning(
                "%s: SYNTHETIC data -- numbers verify the pipeline only", dataset
            )

        for model_name in models:
            run_baseline(config, prepared, model_name, imbalance=args.imbalance)

    if not prepared_any:
        LOGGER.error("Nothing ran")
        return 1
    if not args.no_report:
        build_report(config)
    return 0


def cmd_report(args, config) -> int:
    table = build_report(config)
    print(table.to_string(index=False))
    return 0


def cmd_verify(args, config) -> int:
    ok, messages = verify_artifacts(config, require_models=_models(args.model) or None)
    for message in messages:
        print(message)
    print()
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baselines",
        description="Dataset curation + static LightGBM/XGBoost baselines (RL Idea #4).",
    )
    parser.add_argument("--config", default="configs/baselines.yaml", help="YAML config path")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="curate splits and freeze the preprocessor")
    prepare.add_argument("--dataset", default="all", help="all | healthcare,fraud,network")
    prepare.add_argument("--strict", action="store_true", help="fail instead of using synthetic data")
    prepare.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="train baselines and write results")
    run.add_argument("--dataset", default="all")
    run.add_argument("--model", default="all", help="all | lightgbm,xgboost")
    run.add_argument(
        "--imbalance",
        default=None,
        choices=["class_weight", "smote", "none"],
        help="traditional imbalance handling (default: config value)",
    )
    run.add_argument("--strict", action="store_true", help="fail instead of using synthetic data")
    run.add_argument("--no-report", action="store_true", help="skip the summary report")
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report", help="aggregate results into CSV/MD/figure")
    report.set_defaults(func=cmd_report)

    verify = subparsers.add_parser("verify", help="QA gate over all artefacts")
    verify.add_argument("--model", default="all")
    verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    config.ensure_dirs()
    _setup_logging(config.path("logs_dir"), args.verbose)
    return args.func(args, config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
