"""End-to-end smoke test on the synthetic stand-ins (no download required).

The point is to prove that curation, preprocessing, training, evaluation,
reporting and the QA gate all work together and produce sane artefacts.
"""

import json

import numpy as np
import pytest

from baselines.config import Config
from baselines.pipeline import (
    build_report,
    load_prepared,
    prepare_dataset,
    run_baseline,
    verify_artifacts,
)
from baselines.preprocessing import PreprocessingConfig

pytest.importorskip("lightgbm")

FAST_MODELS = {"lightgbm": {"n_estimators": 40, "num_leaves": 15}}


@pytest.fixture()
def config(tmp_path):
    return Config(
        seed=42,
        models=FAST_MODELS,
        bootstrap_samples=20,
        preprocessing=PreprocessingConfig(),
        root=tmp_path,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        results_dir=tmp_path / "results",
        logs_dir=tmp_path / "logs",
    )


def test_prepare_writes_all_artefacts(config):
    prepared = prepare_dataset(config, "fraud", allow_synthetic=True)
    assert prepared.source == "synthetic"
    assert prepared.schema["task"] == "binary"
    assert prepared.schema["imbalance_ratio"] > 1.0

    out_dir = config.dataset_dir("fraud")
    for name in ("schema.json", "train.parquet", "val.parquet", "test.parquet",
                 "preprocessor.joblib", "arrays.npz"):
        assert (out_dir / name).exists(), name

    reloaded = load_prepared(config, "fraud")
    assert len(reloaded.splits["train"].y) == len(prepared.splits["train"].y)
    assert reloaded.feature_names == prepared.feature_names


def test_run_baseline_reports_valid_metrics(config):
    prepared = prepare_dataset(config, "fraud", allow_synthetic=True)
    result = run_baseline(config, prepared, "lightgbm", imbalance="class_weight")

    metrics = result["metrics"]
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["g_mean"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["n_samples"] == result["n_test_rows"]
    assert result["threshold"] is not None and 0.0 <= result["threshold"] <= 1.0
    # An imbalanced but learnable dataset: the model must beat chance.
    assert metrics["pr_auc"] > 0.5
    assert result["top_features"]

    path = config.path("results_dir") / "fraud__lightgbm.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["dataset"] == "fraud"


def test_multiclass_pipeline_runs(config):
    prepared = prepare_dataset(config, "healthcare", allow_synthetic=True)
    result = run_baseline(config, prepared, "lightgbm")
    assert result["task"] == "multiclass"
    assert result["threshold"] is None
    assert result["metrics"]["n_classes"] == 3
    assert sum(map(sum, result["metrics"]["confusion_matrix"])) == result["n_test_rows"]


def test_report_and_verify_gate(config):
    prepared = prepare_dataset(config, "fraud", allow_synthetic=True)
    run_baseline(config, prepared, "lightgbm")
    table = build_report(config)

    assert len(table) == 1
    assert (config.path("results_dir") / "baseline_summary.csv").exists()
    assert (config.path("results_dir") / "baseline_summary.md").exists()

    ok, messages = verify_artifacts(config, require_models=["lightgbm"], datasets=["fraud"])
    failures = [m for m in messages if m.startswith("FAIL")]
    assert ok, "\n".join(failures)
    assert any("SYNTHETIC" in m for m in messages)
