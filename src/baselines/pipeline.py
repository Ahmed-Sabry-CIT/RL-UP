"""Stage orchestration: curate splits, fit baselines, aggregate, verify.

Public entry points (all also reachable through ``python -m baselines``):

* :func:`prepare_dataset` -- load raw data, stratify, freeze the preprocessor,
  write parquet splits + ``schema.json`` + ``arrays.npz``.
* :func:`run_baseline`   -- train one (dataset, model) baseline and write a
  result JSON with Macro-F1 / G-Mean / PR-AUC (+ bootstrap CIs).
* :func:`build_report`   -- aggregate every result into CSV / Markdown / figures.
* :func:`verify_artifacts` -- QA gate over the produced artefacts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import datasets as ds
from .config import Config
from .metrics import bootstrap_ci, evaluate, select_threshold
from .models import (
    available_models,
    build_model,
    feature_importances,
    fit_model,
    predict_scores,
)
from .preprocessing import TabularPreprocessor
from .splits import SPLIT_NAMES, Split, assert_disjoint, describe, stratified_split

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACTS = ("schema.json", "train.parquet", "val.parquet", "test.parquet")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------- prepare
@dataclass
class PreparedData:
    dataset: str
    task: str
    target_names: list[str]
    positive_class: str | None
    source: str
    splits: dict[str, Split]
    preprocessor: TabularPreprocessor
    schema: dict[str, Any]

    @property
    def labels(self) -> list[int]:
        return list(range(len(self.target_names)))

    @property
    def y(self) -> np.ndarray:
        return np.asarray(self.labels)

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor.feature_names_out_)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception as exc:  # pragma: no cover - depends on pyarrow availability
        fallback = path.with_suffix(".csv")
        logger.warning("Parquet write failed (%s); writing %s instead", exc, fallback.name)
        frame.to_csv(fallback, index=False)
        return fallback


def prepare_dataset(
    config: Config,
    dataset: str,
    allow_synthetic: bool = True,
) -> PreparedData:
    """Curate one dataset and persist every artefact downstream teams need."""
    started = time.perf_counter()
    raw = ds.load(
        dataset,
        config.path("raw_dir"),
        allow_synthetic=allow_synthetic,
        seed=config.seed,
    )

    # Stable, contiguous row identifiers so splits can be proven disjoint.
    raw.X = raw.X.reset_index(drop=True)
    raw.y = raw.y.reset_index(drop=True).astype(int)

    splits = stratified_split(raw.X, raw.y, config.splits, seed=config.seed)
    assert_disjoint(splits)

    preprocessor = TabularPreprocessor(config.preprocessing)
    preprocessor.fit(splits["train"].X)
    logger.info(
        "%s: %d raw features -> %d model features (dropped %d: %s)",
        dataset,
        raw.X.shape[1],
        len(preprocessor.feature_names_out_),
        len(preprocessor.dropped_columns_),
        ", ".join(sorted(preprocessor.dropped_columns_)) or "none",
    )

    out_dir = config.dataset_dir(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in SPLIT_NAMES:
        split = splits[name]
        frame = split.X.copy()
        # Stable positional ids from the curated pool (not re-numbered per
        # split) -- this is what makes disjointness checkable after a reload.
        frame.insert(0, "row_id", split.X.index.to_numpy(dtype=np.int64))
        frame["__target__"] = split.y.to_numpy()
        _write_frame(frame, out_dir / f"{name}.parquet")

    preprocessor.save(out_dir / "preprocessor.joblib")

    arrays = {}
    for name in SPLIT_NAMES:
        arrays[f"{name}_X"] = preprocessor.transform(splits[name].X)
        arrays[f"{name}_y"] = splits[name].y.to_numpy(dtype=np.int64)
    np.savez_compressed(out_dir / "arrays.npz", **arrays)

    schema = {
        "dataset": dataset,
        "source": raw.source,
        "provenance": raw.provenance,
        "task": raw.task,
        "target_names": raw.target_names,
        "labels": raw.labels,
        "positive_class": raw.positive_class,
        "class_counts": raw.class_counts(),
        "imbalance_ratio": round(raw.imbalance_ratio(), 4),
        "split_ratios": config.splits,
        "seed": config.seed,
        "splits": describe(splits, raw.target_names),
        "n_features_raw": int(raw.X.shape[1]),
        "n_features_model": len(preprocessor.feature_names_out_),
        "feature_names": list(preprocessor.feature_names_out_),
        "id_columns": raw.id_columns,
        "dropped_columns": preprocessor.dropped_columns_,
        "preprocessing": preprocessor.config.__dict__,
        "row_id_column": "row_id",
        "target_column": "__target__",
        "created_at": _now(),
        "prepare_seconds": round(time.perf_counter() - started, 3),
    }
    with open(out_dir / "schema.json", "w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)

    logger.info(
        "%s: prepared train/val/test = %s rows -> %s",
        dataset,
        " / ".join(str(len(splits[n].y)) for n in SPLIT_NAMES),
        out_dir,
    )
    return PreparedData(
        dataset=dataset,
        task=raw.task,
        target_names=raw.target_names,
        positive_class=raw.positive_class,
        source=raw.source,
        splits=splits,
        preprocessor=preprocessor,
        schema=schema,
    )


def load_prepared(config: Config, dataset: str) -> PreparedData:
    """Load artefacts written by :func:`prepare_dataset`."""
    out_dir = config.dataset_dir(dataset)
    with open(out_dir / "schema.json", encoding="utf-8") as handle:
        schema = json.load(handle)

    splits: dict[str, Split] = {}
    for name in SPLIT_NAMES:
        path = out_dir / f"{name}.parquet"
        if not path.exists():
            path = path.with_suffix(".csv")
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        y = frame.pop("__target__").astype(int)
        row_ids = frame["row_id"].to_numpy(dtype=np.int64) if "row_id" in frame else None
        frame = frame.drop(columns=["row_id"], errors="ignore")
        if row_ids is not None:
            frame.index = row_ids
            y.index = row_ids
        splits[name] = Split(name, frame, y)

    return PreparedData(
        dataset=dataset,
        task=schema["task"],
        target_names=list(schema["target_names"]),
        positive_class=schema.get("positive_class"),
        source=schema["source"],
        splits=splits,
        preprocessor=TabularPreprocessor.load(out_dir / "preprocessor.joblib"),
        schema=schema,
    )


# -------------------------------------------------------------------- baseline
def _apply_smote(
    X_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Traditional SMOTE oversampling of the training split only."""
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(y_train)
    minority = int(counts[counts > 0].min())
    k_neighbors = max(1, min(5, minority - 1))
    if minority < 2:
        logger.warning("Minority class has <2 rows; skipping SMOTE")
        return X_train, y_train
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


def run_baseline(
    config: Config,
    prepared: PreparedData,
    model_name: str,
    imbalance: str | None = None,
) -> dict[str, Any]:
    """Train and evaluate one static baseline; returns its result record."""
    imbalance = imbalance or config.imbalance
    labels = np.asarray(prepared.labels)
    target_names = prepared.target_names
    started = time.perf_counter()

    X_train = prepared.preprocessor.transform(prepared.splits["train"].X)
    y_train = prepared.splits["train"].y.to_numpy(dtype=int)
    X_val = prepared.preprocessor.transform(prepared.splits["val"].X)
    y_val = prepared.splits["val"].y.to_numpy(dtype=int)
    X_test = prepared.preprocessor.transform(prepared.splits["test"].X)
    y_test = prepared.splits["test"].y.to_numpy(dtype=int)

    if imbalance == "smote":
        X_train, y_train = _apply_smote(X_train, y_train, config.seed)

    params = (config.models or {}).get(model_name, {})
    model = build_model(model_name, params, n_classes=len(labels), seed=config.seed)

    fit_started = time.perf_counter()
    model = fit_model(
        model,
        model_name,
        X_train,
        y_train,
        X_val,
        y_val,
        imbalance=imbalance,
    )
    fit_seconds = time.perf_counter() - fit_started

    val_scores = predict_scores(model, X_val, labels)
    threshold = None
    if len(labels) == 2:
        threshold = select_threshold(y_val, val_scores, labels)
        logger.info("%s/%s: validation-tuned threshold = %.4f", prepared.dataset, model_name, threshold)

    val_metrics = evaluate(y_val, val_scores, labels, target_names, threshold)

    test_scores = predict_scores(model, X_test, labels)
    test_metrics = evaluate(y_test, test_scores, labels, target_names, threshold)

    ci = bootstrap_ci(
        y_test,
        test_scores,
        labels,
        threshold=threshold,
        n_boot=config.bootstrap_samples,
        seed=config.seed,
    )

    result = {
        "dataset": prepared.dataset,
        "model": model_name,
        "data_source": prepared.source,
        "provenance": prepared.schema.get("provenance"),
        "task": prepared.task,
        "target_names": target_names,
        "positive_class": prepared.positive_class,
        "imbalance_strategy": imbalance,
        "seed": config.seed,
        "n_features_raw": prepared.schema.get("n_features_raw"),
        "n_features_model": len(prepared.feature_names),
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(y_val)),
        "n_test_rows": int(len(y_test)),
        "threshold": threshold,
        "params": dict(params),
        "metrics": test_metrics.to_dict(),
        "validation_metrics": val_metrics.to_dict(),
        "top_features": feature_importances(model, prepared.feature_names, top_k=20),
        "timings_seconds": {
            "fit": round(fit_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "run_at": _now(),
    }
    result["metrics"]["ci"] = ci

    results_path = config.path("results_dir") / f"{prepared.dataset}__{model_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    logger.info(
        "%s/%s: Macro-F1 %.4f | G-Mean %.4f | PR-AUC %.4f | n_test=%d -> %s",
        prepared.dataset,
        model_name,
        test_metrics.macro_f1,
        test_metrics.g_mean,
        test_metrics.pr_auc,
        test_metrics.n_samples,
        results_path.name,
    )
    return result


def collect_results(config: Config) -> list[dict[str, Any]]:
    results = []
    for path in sorted(config.path("results_dir").glob("*__*.json")):
        with open(path, encoding="utf-8") as handle:
            results.append(json.load(handle))
    return results


# ---------------------------------------------------------------------- report
def build_report(config: Config) -> pd.DataFrame:
    """Aggregate results into ``baseline_summary.{csv,md}`` plus a figure."""
    results = collect_results(config)
    if not results:
        raise RuntimeError("No baseline results found; run `python -m baselines run` first")

    rows = []
    for record in results:
        metrics = record["metrics"]
        ci = metrics.get("ci", {})
        rows.append(
            {
                "dataset": record["dataset"],
                "model": record["model"],
                "data_source": record["data_source"],
                "task": record["task"],
                "n_train": record["n_train_rows"],
                "n_test": record["n_test_rows"],
                "n_features_model": record["n_features_model"],
                "macro_f1": round(metrics["macro_f1"], 4),
                "macro_f1_ci": _fmt_ci(ci.get("macro_f1")),
                "g_mean": round(metrics["g_mean"], 4),
                "g_mean_ci": _fmt_ci(ci.get("g_mean")),
                "pr_auc": round(metrics["pr_auc"], 4),
                "pr_auc_ci": _fmt_ci(ci.get("pr_auc")),
                "accuracy": round(metrics["accuracy"], 4),
                # multiclass baselines use argmax, so there is no threshold
                "threshold": (
                    round(record["threshold"], 6)
                    if record["threshold"] is not None
                    else "n/a (argmax)"
                ),
                "fit_seconds": record["timings_seconds"]["fit"],
            }
        )

    table = pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    results_dir = config.path("results_dir")
    table.to_csv(results_dir / "baseline_summary.csv", index=False)
    (results_dir / "baseline_summary.md").write_text(
        "# Static baseline results\n\n" + table.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    _make_figure(table, results_dir / "baseline_metrics.png")
    logger.info("Wrote %s", results_dir / "baseline_summary.md")
    return table


def _fmt_ci(interval: list[float] | None) -> str:
    if not interval:
        return ""
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def _make_figure(table: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("matplotlib unavailable; skipping %s", path.name)
        return

    metrics = ["macro_f1", "g_mean", "pr_auc"]
    datasets = list(dict.fromkeys(table["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8), squeeze=False)
    width = 0.8 / max(table["model"].nunique(), 1)

    for ax, dataset in zip(axes[0], datasets):
        subset = table[table["dataset"] == dataset]
        positions = np.arange(len(metrics))
        for offset, (_, row) in enumerate(subset.iterrows()):
            values = [row[m] for m in metrics]
            bars = ax.bar(positions + offset * width, values, width, label=row["model"])
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_xticks(positions + width * (len(subset) - 1) / 2)
        ax.set_xticklabels(["Macro-F1", "G-Mean", "PR-AUC"], fontsize=8)
        ax.set_ylim(0, 1.12)
        source = subset["data_source"].iloc[0]
        ax.set_title(f"{dataset} ({source})", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Static baselines: uncoupled preprocessing (higher is better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------- verify
def verify_artifacts(
    config: Config,
    require_models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """QA gate over the produced artefacts. Returns (ok, messages)."""
    messages: list[str] = []
    failures = 0

    def check(condition: bool, message: str) -> None:
        nonlocal failures
        messages.append(("PASS  " if condition else "FAIL  ") + message)
        if not condition:
            failures += 1

    for dataset in list(datasets or ds.DATASETS):
        out_dir = config.dataset_dir(dataset)
        for artifact in REQUIRED_ARTIFACTS:
            path = out_dir / artifact
            if not path.exists() and artifact.endswith(".parquet"):
                path = path.with_suffix(".csv")
            check(path.exists(), f"{dataset}: artefact {path.name} exists")
        if not (out_dir / "schema.json").exists():
            continue

        with open(out_dir / "schema.json", encoding="utf-8") as handle:
            schema = json.load(handle)
        check(
            schema["source"] in {"real", "synthetic"},
            f"{dataset}: data source recorded ({schema['source']})",
        )
        if schema["source"] == "synthetic":
            messages.append(
                "WARN  "
                f"{dataset}: SYNTHETIC data -- metrics are a pipeline check, not benchmarks"
            )

        splits = load_prepared(config, dataset).splits
        try:
            assert_disjoint(splits)
            check(True, f"{dataset}: splits are disjoint")
        except AssertionError as exc:
            check(False, f"{dataset}: {exc}")

        n_classes = len(schema["target_names"])
        for name, split in splits.items():
            present = set(split.y.unique().tolist())
            check(
                present and present <= set(range(n_classes)),
                f"{dataset}/{name}: labels within [0, {n_classes - 1}]",
            )
            check(
                len(present) == n_classes,
                f"{dataset}/{name}: all {n_classes} classes present",
            )

        expected = {"train": config.splits["train"], "val": config.splits["val"], "test": config.splits["test"]}
        total = sum(len(s.y) for s in splits.values())
        for name, target in expected.items():
            observed = len(splits[name].y) / total
            check(
                abs(observed - target) < 0.02,
                f"{dataset}/{name}: share {observed:.3f} within 0.02 of {target}",
            )

        check(
            schema["n_features_model"] == len(schema["feature_names"]),
            f"{dataset}: feature-name count matches n_features_model",
        )

    results = collect_results(config)
    check(bool(results), "results: at least one baseline result exists")
    models_seen = {r["model"] for r in results}
    if require_models:
        for model in require_models:
            check(model in models_seen, f"results: {model} baseline present")

    for record in results:
        tag = f"{record['dataset']}/{record['model']}"
        metrics = record["metrics"]
        check(metrics["n_samples"] == record["n_test_rows"], f"{tag}: n_samples matches test rows")
        for key in ("macro_f1", "g_mean", "pr_auc", "accuracy"):
            value = metrics.get(key)
            check(
                isinstance(value, (int, float)) and np.isfinite(value) and 0.0 <= value <= 1.0,
                f"{tag}: {key} in [0, 1] ({value:.4f})" if isinstance(value, (int, float)) else f"{tag}: {key} numeric",
            )
        matrix = metrics.get("confusion_matrix") or []
        flat_sum = sum(sum(row) for row in matrix)
        check(flat_sum == metrics["n_samples"], f"{tag}: confusion matrix sums to n_samples")
        check(
            len(matrix) == len(record["target_names"]) and all(len(r) == len(matrix) for r in matrix),
            f"{tag}: confusion matrix is square over target classes",
        )
        if record["task"] == "binary":
            thr = record["threshold"]
            check(isinstance(thr, float) and 0.0 <= thr <= 1.0, f"{tag}: threshold in [0, 1]")

    results_dir = config.path("results_dir")
    check((results_dir / "baseline_summary.csv").exists(), "report: summary CSV exists")
    check((results_dir / "baseline_summary.md").exists(), "report: summary Markdown exists")

    return failures == 0, messages
