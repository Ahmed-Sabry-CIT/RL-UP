"""Unit tests for the shared metric definitions."""

import numpy as np
import pytest

from baselines.metrics import (
    bootstrap_ci,
    evaluate,
    g_mean,
    macro_f1,
    pr_auc,
    predict_from_scores,
    select_threshold,
)

LABELS = [0, 1]
NAMES = ["neg", "pos"]


def test_g_mean_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1])
    assert g_mean(y, y, LABELS) == pytest.approx(1.0)
    # Predicting the majority class only -> one recall is zero -> G-Mean 0.
    assert g_mean(y, np.zeros_like(y), LABELS) == pytest.approx(0.0)


def test_g_mean_equals_sqrt_of_sensitivity_times_specificity():
    # positives: [1,1,1,1] predicted [1,1,0,0]  -> recall 0.5
    # negatives: [0,0,0,0] predicted [0,0,0,0]  -> recall 1.0
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    assert g_mean(y_true, y_pred, LABELS) == pytest.approx(np.sqrt(0.5 * 1.0))


def test_macro_f1_hand_computed():
    # class 0: TP=1, FP=0, FN=1 -> F1 = 2/3 ; class 1: TP=2, FP=1, FN=0 -> F1 = 0.8
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert macro_f1(y_true, y_pred, LABELS) == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_multiclass_penalises_missing_class():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 2])
    perfect = macro_f1(y_true, y_pred, [0, 1, 2])
    y_pred_bad = np.array([0, 1, 1, 1])
    assert perfect == pytest.approx(1.0)
    assert macro_f1(y_true, y_pred_bad, [0, 1, 2]) < perfect


def test_pr_auc_perfect_separation_is_one():
    y = np.array([0, 0, 1, 1])
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    value, per_class = pr_auc(y, scores, LABELS)
    assert value == pytest.approx(1.0)
    assert per_class["0"] == pytest.approx(1.0)
    assert per_class["1"] == pytest.approx(1.0)


def test_predict_from_scores_uses_threshold_for_binary():
    scores = np.array([[0.4, 0.6], [0.7, 0.3]])
    assert predict_from_scores(scores, LABELS, 0.5).tolist() == [1, 0]
    assert predict_from_scores(scores, LABELS, 0.2).tolist() == [1, 1]
    # argmax when no threshold is passed
    assert predict_from_scores(scores, LABELS, None).tolist() == [1, 0]


def test_select_threshold_prefers_separating_threshold():
    y = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.85, 0.15], [0.4, 0.6]]
    )
    threshold = select_threshold(y, scores, LABELS)
    assert 0.2 < threshold <= 0.6
    assert macro_f1(y, predict_from_scores(scores, LABELS, threshold), LABELS) == pytest.approx(1.0)


def test_evaluate_returns_consistent_bundle():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    scores = rng.random((300, 3))
    scores = scores / scores.sum(axis=1, keepdims=True)
    metrics = evaluate(y, scores, [0, 1, 2], ["a", "b", "c"])

    assert metrics.n_samples == 300
    assert metrics.n_classes == 3
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.g_mean <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0
    assert sum(map(sum, metrics.confusion_matrix)) == 300
    assert set(metrics.per_class_support) == {"a", "b", "c"}

    payload = metrics.to_dict()
    assert payload["macro_f1"] == metrics.macro_f1


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    # strong, correctly-oriented signal so the interval is tight
    scores = np.column_stack(
        [rng.normal(0, 1, 200) - y * 3.0, rng.normal(0, 1, 200) + y * 3.0]
    )
    # The threshold must be the same one the point estimate uses: with
    # threshold=None a two-column score matrix is read with argmax.
    ci = bootstrap_ci(y, scores, LABELS, threshold=0.5, n_boot=200, seed=3)
    point = evaluate(y, scores, LABELS, NAMES, threshold=0.5)
    assert ci["macro_f1"][0] <= point.macro_f1 <= ci["macro_f1"][1]
    assert ci["g_mean"][0] <= point.g_mean <= ci["g_mean"][1]
    assert ci["pr_auc"][0] <= point.pr_auc <= ci["pr_auc"][1]
