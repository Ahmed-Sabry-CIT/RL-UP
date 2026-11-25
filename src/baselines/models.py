"""Static gradient-boosting baselines: LightGBM and XGBoost.

Both models are trained with *traditional* imbalance handling -- balanced class
weights (or, optionally, SMOTE applied to the training split only).  Neither
model sees any TreeSHAP-guided feature selection or adaptive augmentation; that
is the MARL contribution this baselines module exists to measure against.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAMES = ("lightgbm", "xgboost")


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights (sklearn's ``balanced`` convention)."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = y.size / (classes.size * counts)
    mapping = {int(c): float(w) for c, w in zip(classes, weight_per_class)}
    return np.array([mapping[int(c)] for c in y], dtype=np.float64)


def available_models() -> list[str]:
    """Which of the supported libraries are importable in this environment."""
    found = []
    for name in MODEL_NAMES:
        try:
            __import__("lightgbm" if name == "lightgbm" else "xgboost")
            found.append(name)
        except Exception:  # pragma: no cover - depends on environment
            logger.warning("Model backend %r is not installed; it will be skipped", name)
    return found


def build_model(
    name: str,
    params: dict[str, Any] | None,
    n_classes: int,
    seed: int = 42,
):
    """Construct an untrained estimator for ``name``.

    Class weighting is deliberately *not* set on the estimator: it is applied
    once, as ``sample_weight`` in :func:`fit_model`, so LightGBM and XGBoost
    are weighted identically (setting both would square the weights and push
    the model far too hard towards the minority class).
    """
    params = dict(params or {})
    objective = "binary" if n_classes == 2 else "multiclass"

    if name == "lightgbm":
        import lightgbm as lgb

        default = dict(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        default.update(params)
        return lgb.LGBMClassifier(objective=objective, **default)

    if name == "xgboost":
        from xgboost import XGBClassifier

        default = dict(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=seed,
            tree_method="hist",
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            verbosity=0,
        )
        default.update(params)
        return XGBClassifier(
            objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            **default,
        )

    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")


def fit_model(
    model,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    imbalance: str = "class_weight",
    early_stopping_rounds: int = 50,
) -> Any:
    """Fit ``model``, using the validation split for early stopping."""
    sample_weight = (
        balanced_sample_weights(y_train) if imbalance == "class_weight" else None
    )
    eval_set = [(X_val, y_val)] if X_val is not None and len(y_val) > 0 else None

    if name == "lightgbm":
        import lightgbm as lgb

        callbacks = []
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            callbacks.append(
                lgb.early_stopping(early_stopping_rounds, verbose=False)
            )
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["eval_metric"] = "binary_logloss" if len(np.unique(y_train)) == 2 else "multi_logloss"
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            callbacks=callbacks,
            **fit_kwargs,
        )
        return model

    if name == "xgboost":
        import inspect

        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False
            # xgboost >= 2.0 moved early stopping from fit() to the constructor
            if "early_stopping_rounds" in inspect.signature(model.fit).parameters:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            else:
                model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)
        return model

    raise ValueError(f"Unknown model {name!r}")


def predict_scores(model, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Probability matrix with columns aligned to ``labels``.

    Estimators order ``predict_proba`` columns by ``model.classes_``, which can
    differ from the label order used for evaluation, so realign explicitly.
    """
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(getattr(model, "classes_", labels))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])

    out = np.zeros((proba.shape[0], len(labels)), dtype=float)
    for source_index, cls in enumerate(classes):
        matches = np.where(labels == cls)[0]
        if matches.size:
            out[:, matches[0]] = proba[:, source_index]
    return out


def feature_importances(model, feature_names: list[str], top_k: int = 20) -> list[dict]:
    """Top-k importances, ordered descending (used by the XAI track)."""
    raw = getattr(model, "feature_importances_", None)
    if raw is None:
        return []
    raw = np.asarray(raw, dtype=float)
    if raw.size != len(feature_names):
        return []
    order = np.argsort(raw)[::-1][:top_k]
    total = float(raw.sum()) or 1.0
    return [
        {
            "feature": feature_names[i],
            "importance": float(raw[i]),
            "importance_share": float(raw[i] / total),
        }
        for i in order
    ]
