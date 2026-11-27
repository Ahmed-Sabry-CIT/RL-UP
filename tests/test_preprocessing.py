"""Tests for the traditional, uncoupled preprocessor."""

import numpy as np
import pandas as pd

from baselines.preprocessing import PreprocessingConfig, TabularPreprocessor


def _frame(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "num_missing": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "cat": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice([True, False], n),
            "all_nan": np.nan,
            "mostly_missing": np.where(rng.random(n) < 0.9, np.nan, 1.0),
            "id_like": [f"row-{i}" for i in range(n)],
        }
    )


def test_drops_unusable_columns():
    frame = _frame()
    pre = TabularPreprocessor().fit(frame)
    assert "all_nan" in pre.dropped_columns_
    assert "mostly_missing" in pre.dropped_columns_
    assert "id_like" in pre.dropped_columns_
    assert "num" in pre.numeric_columns_
    assert "cat" in pre.categorical_columns_


def test_output_is_finite_dense_float32():
    frame = _frame()
    pre = TabularPreprocessor().fit(frame)
    matrix = pre.transform(frame)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert matrix.shape == (len(frame), len(pre.feature_names_out_))


def test_missing_indicator_column_is_created():
    frame = _frame()
    pre = TabularPreprocessor().fit(frame)
    assert "num_missing__is_missing" in pre.feature_names_out_


def test_unseen_category_does_not_crash():
    train = _frame()
    pre = TabularPreprocessor().fit(train)
    test = pd.DataFrame({"num": [0.0], "num_missing": [np.nan], "cat": ["never-seen"], "flag": [True]})
    out = pre.transform(test)
    assert out.shape[0] == 1


def test_transform_before_fit_raises():
    pre = TabularPreprocessor()
    try:
        pre.transform(_frame(5))
    except RuntimeError as exc:
        assert "before fit" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


def test_scaling_is_fit_on_train_only():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"num": rng.normal(10, 1, 500)})
    test = pd.DataFrame({"num": rng.normal(1000, 1, 500)})
    pre = TabularPreprocessor(PreprocessingConfig(scale=True)).fit(train)
    train_matrix = pre.transform(train)
    test_matrix = pre.transform(test)
    # train is standardised around zero; test stays far away (no refitting)
    assert abs(float(train_matrix.mean())) < 0.05
    assert float(test_matrix.mean()) > 50


def test_scaler_can_be_disabled_and_round_trips():
    import tempfile
    from pathlib import Path

    frame = _frame(50)
    pre = TabularPreprocessor(PreprocessingConfig(scale=False)).fit(frame)
    with tempfile.TemporaryDirectory() as tmp:
        path = pre.save(Path(tmp) / "pre.joblib")
        restored = TabularPreprocessor.load(path)
    np.testing.assert_allclose(pre.transform(frame), restored.transform(frame))
    assert restored.feature_names_out_ == pre.feature_names_out_
