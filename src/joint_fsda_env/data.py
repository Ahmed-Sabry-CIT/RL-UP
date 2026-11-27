"""Deterministic synthetic demo data for offline development and tests.

Real experiments feed ``JointFSDAEnv`` from the curated splits produced by the
dataset component (``data/processed/<dataset>/``, see ``docs/BASELINES.md``).
This module only exists so the environment, the reward-calibration script and
the test-suite run without the ~170 MB benchmarks or network access.
"""
from __future__ import annotations

import numpy as np

MINORITY_LABEL = 1  # the environment requires label 1 to be the minority class

# Four minority sub-clusters of decreasing size (KMeans should recover them),
# expressed in a 3-D latent space before projection onto the feature space.
_MINORITY_MODES = np.array([[-1.6, 0.0, 0.0],
                            [0.0, 1.6, 0.0],
                            [1.5, -1.3, 0.0],
                            [0.0, -1.4, 1.5]])
_MINORITY_SHARES = np.array([0.4, 0.3, 0.2, 0.1])
_MAJORITY_MODES = np.array([[-0.9, 0.0, 0.0], [0.7, 0.0, 0.0]])


def make_demo_data(n=1500, seed=0, n_features=20, minority_frac=0.12, val_frac=0.2):
    """Return ``(X_train, y_train, X_val, y_val)``: float64 features, int labels.

    The minority class (label 1) is a mixture of four Gaussian sub-clusters so the
    environment's KMeans step has reproducible, non-trivial structure; features
    ``8:12`` are noisy copies of ``0:4`` to give the feature-selection agent
    redundant columns to prune. Features are standardised, labels are binary and
    the train/validation split is stratified.
    """
    if n < 50:
        raise ValueError("n must be >= 50 to build both splits")
    if not 0.0 < minority_frac < 0.5:
        raise ValueError("minority_frac must be in (0, 0.5)")
    rng = np.random.default_rng(seed)

    n_min = max(8, int(round(n * minority_frac)))
    n_maj = n - n_min

    counts = np.floor(_MINORITY_SHARES * n_min).astype(int)
    counts[-1] += n_min - counts.sum()
    minors = np.vstack([mode + 0.55 * rng.normal(size=(k, 3))
                        for mode, k in zip(_MINORITY_MODES, counts)])
    majors = (_MAJORITY_MODES[rng.integers(0, 2, size=n_maj)]
              + 0.9 * rng.normal(size=(n_maj, 3)))
    z = np.vstack([majors, minors])
    y = np.concatenate([np.zeros(n_maj, int), np.full(n_min, MINORITY_LABEL)])

    loadings = rng.normal(0.0, 1.0, size=(3, n_features)) / np.sqrt(3.0)
    X = z @ loadings + 0.35 * rng.normal(size=(len(y), n_features))
    if n_features >= 12:
        X[:, 8:12] = 0.9 * X[:, 0:4] + 0.1 * rng.normal(size=(len(y), 4))
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    train_idx, val_idx = _stratified_split(y, val_frac, rng)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _stratified_split(y, val_frac, rng):
    """Per-class shuffle so both splits keep the minority class."""
    train_idx, val_idx = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * val_frac)))
        val_idx.append(idx[:cut])
        train_idx.append(idx[cut:])
    return np.concatenate(train_idx), np.concatenate(val_idx)
