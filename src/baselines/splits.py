"""Curated, reproducible train / validation / test splits.

Splits are stratified on the target so rare classes survive in every split, and
every row keeps a stable ``row_id`` so overlap can be asserted in the tests and
in ``baselines verify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class Split:
    name: str
    X: pd.DataFrame
    y: pd.Series


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
) -> dict[str, Split]:
    """Return ``{"train": ..., "val": ..., "test": ...}`` with disjoint indices."""
    ratios = dict(ratios or {"train": 0.6, "val": 0.2, "test": 0.2})
    missing = {"train", "val"} - set(ratios)
    if missing:
        raise ValueError(f"Split ratios must define train and val; missing {missing}")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios.values()):.4f}")

    # Positional indices are the stable row identifiers: keeping them (instead
    # of re-indexing each subset from 0) is what makes overlap detectable.
    X = X.reset_index(drop=True).copy()
    y = y.reset_index(drop=True).copy()
    y = y.astype(int)

    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Every class needs at least 2 samples for a stratified split; "
            f"got counts {counts.to_dict()}"
        )

    test_fraction = ratios.get("test", 0.0)
    val_share = ratios["val"] / max(ratios["train"] + ratios["val"], 1e-9)

    if test_fraction > 0:
        X_rest, X_test, y_rest, y_test = train_test_split(
            X, y, test_size=test_fraction, random_state=seed, stratify=y
        )
    else:
        X_rest, X_test, y_rest, y_test = X, X.iloc[0:0], y, y.iloc[0:0]

    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share, random_state=seed, stratify=y_rest
    )

    result = {
        "train": Split("train", X_train, y_train),
        "val": Split("val", X_val, y_val),
        "test": Split("test", X_test, y_test),
    }
    _validate(result, y.nunique())
    return result


def _validate(splits: dict[str, Split], n_classes: int) -> None:
    seen: set[int] = set()
    for split_name, split in splits.items():
        if len(split.X) == 0:
            continue
        present = sorted(split.y.unique().tolist())
        if len(present) < min(n_classes, 2):
            raise ValueError(
                f"Split {split_name!r} contains {len(present)} class(es) {present}; "
                "increase the dataset size or lower the number of classes"
            )
        overlap = seen & set(split.X.index.tolist())
        if overlap:
            raise ValueError(f"Split {split_name!r} overlaps previous splits")
        seen |= set(split.X.index.tolist())


def describe(splits: dict[str, Split], target_names: list[str]) -> dict:
    """Class coverage per split, for the schema file and the logs."""
    out: dict[str, dict] = {}
    for name, split in splits.items():
        counts = np.bincount(split.y.to_numpy(dtype=int), minlength=len(target_names))
        out[name] = {
            "n_rows": int(len(split.y)),
            "class_counts": {
                target_names[i]: int(counts[i]) for i in range(len(target_names))
            },
            "class_fractions": {
                target_names[i]: round(float(counts[i] / max(len(split.y), 1)), 6)
                for i in range(len(target_names))
            },
        }
    return out


def assert_disjoint(splits: dict[str, Split]) -> None:
    """Raise if any row identifier appears in more than one split."""
    owner: dict[int, str] = {}
    for name, split in splits.items():
        for row in split.X.index.tolist():
            if row in owner:
                raise AssertionError(f"row {row} in both {owner[row]!r} and {name!r}")
            owner[row] = name
