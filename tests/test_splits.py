"""Tests for the curated stratified splits."""

import numpy as np
import pandas as pd
import pytest

from baselines.splits import assert_disjoint, describe, stratified_split


def _toy(n=1200, seed=0, n_classes=3):
    rng = np.random.default_rng(seed)
    weights = [0.7, 0.2, 0.1][:n_classes]
    y = pd.Series(rng.choice(n_classes, size=n, p=np.array(weights) / sum(weights)))
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.integers(0, 5, n)})
    return X, y


def test_split_sizes_and_disjointness():
    X, y = _toy()
    splits = stratified_split(X, y, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=42)
    total = len(X)
    assert len(splits["train"].y) == pytest.approx(0.6 * total, abs=3)
    assert len(splits["val"].y) == pytest.approx(0.2 * total, abs=3)
    assert len(splits["test"].y) == pytest.approx(0.2 * total, abs=3)
    assert sum(len(s.y) for s in splits.values()) == total
    assert_disjoint(splits)


def test_split_is_stratified():
    X, y = _toy(n=2000)
    splits = stratified_split(X, y, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=7)
    overall = y.value_counts(normalize=True).sort_index()
    for split in splits.values():
        shares = split.y.value_counts(normalize=True).sort_index()
        assert set(shares.index) == set(overall.index)
        for label in overall.index:
            assert abs(shares[label] - overall[label]) < 0.02


def test_split_is_reproducible():
    X, y = _toy()
    a = stratified_split(X, y, seed=11)
    b = stratified_split(X, y, seed=11)
    for name in ("train", "val", "test"):
        pd.testing.assert_frame_equal(a[name].X, b[name].X)


def test_no_leakage_between_splits():
    X, y = _toy()
    splits = stratified_split(X, y, seed=3)
    train_ids = set(splits["train"].X.index)
    test_ids = set(splits["test"].X.index)
    assert not (train_ids & test_ids)
    assert_disjoint(splits)


def test_ratios_must_sum_to_one():
    X, y = _toy()
    with pytest.raises(ValueError):
        stratified_split(X, y, {"train": 0.6, "val": 0.2, "test": 0.3}, seed=1)


def test_single_sample_class_is_rejected():
    X = pd.DataFrame({"f": np.arange(10)})
    y = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    with pytest.raises(ValueError):
        stratified_split(X, y, seed=1)


def test_describe_reports_counts_and_fractions():
    X, y = _toy(n=300)
    splits = stratified_split(X, y, seed=5)
    summary = describe(splits, ["a", "b", "c"])
    assert summary["train"]["n_rows"] == len(splits["train"].y)
    train_share = sum(summary["train"]["class_fractions"].values())
    assert train_share == pytest.approx(1.0, abs=1e-4)
