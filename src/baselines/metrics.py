"""Evaluation metrics shared by the baseline pipelines.

The MARL track (Idea #4) reports three headline metrics, fixed here once so the
baseline tables and the multi-agent tables are directly comparable:

* Macro-F1  -- unweighted mean of per-class F1 (penalises ignoring rare classes).
* G-Mean    -- geometric mean of per-class recall; 0.0 if any class is never
               recovered.  For binary problems this is exactly
               sqrt(TPR * TNR) (Kubat & Matwin).
* PR-AUC    -- threshold-free average precision.  Binary: average precision of
               the positive class.  Multiclass: macro one-vs-rest average
               precision.

Predictions are obtained from a probability threshold (binary) or argmax
(multiclass).  For binary problems the threshold is tuned on the validation
split and then frozen for the test split, so test metrics are never leaked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


@dataclass
class Metrics:
    """Container for one evaluation result (JSON-serialisable)."""

    macro_f1: float
    g_mean: float
    pr_auc: float
    accuracy: float
    n_samples: int
    n_classes: int
    per_class_recall: dict[str, float] = field(default_factory=dict)
    per_class_f1: dict[str, float] = field(default_factory=dict)
    per_class_support: dict[str, int] = field(default_factory=dict)
    pr_auc_per_class: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    threshold: float | None = None
    ci: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_f1": self.macro_f1,
            "g_mean": self.g_mean,
            "pr_auc": self.pr_auc,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
            "pr_auc_per_class": self.pr_auc_per_class,
            "confusion_matrix": self.confusion_matrix,
            "threshold": self.threshold,
            "ci": self.ci,
        }


def g_mean(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    """Geometric mean of per-class recall (0.0 if a class is never recovered)."""
    labels = list(labels)
    recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recalls = np.asarray(recalls, dtype=float)
    if recalls.size == 0 or np.any(recalls <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(recalls))))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
    )


def pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[int],
) -> tuple[float, dict[str, float]]:
    """Macro one-vs-rest average precision.

    ``y_score`` has shape (n_samples, n_classes) with columns ordered like
    ``labels``.
    """
    labels = list(labels)
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    per_class: dict[str, float] = {}
    values: list[float] = []
    for idx, label in enumerate(labels):
        binary = (np.asarray(y_true) == label).astype(int)
        if binary.sum() == 0 or binary.sum() == binary.size:
            # Undefined AP for an absent / single-class label; skip from the mean.
            continue
        ap = float(average_precision_score(binary, scores[:, idx]))
        per_class[str(label)] = ap
        values.append(ap)
    return (float(np.mean(values)) if values else 0.0), per_class


def predict_from_scores(
    y_score: np.ndarray,
    labels: Sequence[int],
    threshold: float | None = None,
) -> np.ndarray:
    """Turn class scores into hard labels."""
    labels = np.asarray(list(labels))
    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if threshold is not None and len(labels) == 2:
        return np.where(scores[:, 1] >= threshold, labels[1], labels[0])
    return labels[np.argmax(scores, axis=1)]


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[int],
    max_candidates: int = 512,
) -> float:
    """Pick the binary decision threshold maximising validation Macro-F1.

    Ties are broken by G-Mean, then by the threshold closest to 0.5.  Only
    meaningful for binary problems.
    """
    labels = list(labels)
    if len(labels) != 2:
        raise ValueError("select_threshold is only defined for binary problems")
    scores = np.asarray(y_score, dtype=float)[:, 1]
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, max_candidates)))

    best_thr, best_key = 0.5, (-1.0, -1.0, -1.0)
    for thr in candidates:
        pred = predict_from_scores(y_score, labels, float(thr))
        key = (
            macro_f1(y_true, pred, labels),
            g_mean(y_true, pred, labels),
            -abs(float(thr) - 0.5),
        )
        if key > best_key:
            best_thr, best_key = float(thr), key
    return best_thr


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[int],
    target_names: Sequence[str],
    threshold: float | None = None,
) -> Metrics:
    """Compute the full metric bundle for one prediction set."""
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_pred = predict_from_scores(y_score, labels, threshold)

    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    supports = [int((y_true == lab).sum()) for lab in labels]

    ap, ap_per_class = pr_auc(y_true, y_score, labels)

    return Metrics(
        macro_f1=macro_f1(y_true, y_pred, labels),
        g_mean=g_mean(y_true, y_pred, labels),
        pr_auc=ap,
        accuracy=float((y_true == y_pred).mean()),
        n_samples=int(y_true.size),
        n_classes=len(labels),
        per_class_recall={
            target_names[i]: float(recalls[i]) for i in range(len(labels))
        },
        per_class_f1={target_names[i]: float(f1s[i]) for i in range(len(labels))},
        per_class_support={
            target_names[i]: supports[i] for i in range(len(labels))
        },
        pr_auc_per_class=ap_per_class,
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        threshold=threshold,
    )


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Sequence[int],
    threshold: float | None = None,
    n_boot: int = 200,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, list[float]]:
    """Percentile bootstrap confidence intervals for Macro-F1, G-Mean, PR-AUC.

    The bootstrap is *stratified*: each class is resampled with replacement
    inside itself, so the class priors (and therefore the imbalance ratio) are
    held fixed.  With plain resampling, rare classes drop out of a resample and
    the interval skews away from the point estimate -- exactly the failure mode
    that matters on imbalanced data.
    """
    if n_boot <= 0:
        return {}
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)

    class_indices = [np.where(y_true == label)[0] for label in np.unique(y_true)]
    class_indices = [idx for idx in class_indices if idx.size > 1]
    if not class_indices:
        return {}

    samples: dict[str, list[float]] = {"macro_f1": [], "g_mean": [], "pr_auc": []}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(part, size=part.size, replace=True) for part in class_indices]
        )
        yt, ys = y_true[idx], y_score[idx]
        if np.unique(yt).size < 2:
            continue
        pred = predict_from_scores(ys, labels, threshold)
        samples["macro_f1"].append(macro_f1(yt, pred, labels))
        samples["g_mean"].append(g_mean(yt, pred, labels))
        samples["pr_auc"].append(pr_auc(yt, ys, labels)[0])

    ci: dict[str, list[float]] = {}
    for key, values in samples.items():
        if not values:
            continue
        lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        ci[key] = [float(lo), float(hi)]
    return ci
