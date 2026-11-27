"""Traditional, *uncoupled* tabular preprocessing.

This is deliberately a single static transform fitted on the training split
only (no feedback loop with feature selection or data augmentation -- that
coupling is what the MARL track introduces):

1. drop unusable columns (all-missing, identifier-like, too many missings,
   object columns with excessive cardinality);
2. median / most-frequent imputation, plus missing-value indicator columns;
3. standard scaling of numeric features;
4. one-hot encoding of categorical features, grouping rare levels (so a
   moderately high-cardinality column such as NSL-KDD's ``service`` is kept
   rather than thrown away) with unseen categories ignored.

The fitted object is serialised so the agent developers can reuse the exact
same feature space and column order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingConfig:
    max_missing_fraction: float = 0.45
    max_cardinality: int = 100
    min_frequency: float = 0.01
    scale: bool = True
    add_missing_indicators: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PreprocessingConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TabularPreprocessor:
    """Fit-on-train transform producing a dense float32 matrix."""

    config: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    numeric_columns_: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)
    dropped_columns_: dict[str, str] = field(default_factory=dict)
    feature_names_out_: list[str] = field(default_factory=list)

    _numeric_fill: dict[str, float] = field(default_factory=dict)
    _categorical_fill: dict[str, str] = field(default_factory=dict)
    _missing_indicator_columns: list[str] = field(default_factory=list)
    _scaler: StandardScaler | None = None
    _onehot: OneHotEncoder | None = None
    _fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        X = X.copy()
        self.numeric_columns_, self.categorical_columns_ = [], []
        self.dropped_columns_ = {}
        self._numeric_fill, self._categorical_fill = {}, {}
        self._missing_indicator_columns = []

        n_rows = max(len(X), 1)
        for column in X.columns:
            series = X[column]
            missing_fraction = float(series.isna().mean())

            if series.isna().all():
                self.dropped_columns_[column] = "all_missing"
                continue
            if missing_fraction > self.config.max_missing_fraction:
                self.dropped_columns_[column] = f"missing_fraction={missing_fraction:.3f}"
                continue

            if pd.api.types.is_bool_dtype(series):
                self.numeric_columns_.append(column)
                self._numeric_fill[column] = float(series.mode(dropna=True).iloc[0])
                continue

            if pd.api.types.is_numeric_dtype(series):
                self.numeric_columns_.append(column)
                fill = series.median()
                self._numeric_fill[column] = float(0.0 if pd.isna(fill) else fill)
                if self.config.add_missing_indicators and series.isna().any():
                    self._missing_indicator_columns.append(column)
                continue

            # object / category / string
            unique_ratio = series.nunique(dropna=True) / n_rows
            cardinality = int(series.nunique(dropna=True))
            if unique_ratio > 0.5 and cardinality > self.config.max_cardinality:
                self.dropped_columns_[column] = "identifier_like"
                continue
            if cardinality > self.config.max_cardinality:
                self.dropped_columns_[column] = f"high_cardinality={cardinality}"
                continue

            self.categorical_columns_.append(column)
            mode = series.astype("object").mode(dropna=True)
            self._categorical_fill[column] = str(mode.iloc[0]) if len(mode) else "missing"

        if self.numeric_columns_:
            numeric_block = self._impute_numeric(X[self.numeric_columns_])
            if self.config.scale:
                self._scaler = StandardScaler()
                self._scaler.fit(numeric_block)

        if self.categorical_columns_:
            categorical_block = self._impute_categorical(X[self.categorical_columns_])
            self._onehot = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=self.config.min_frequency or None,
                sparse_output=False,
                dtype=np.float32,
            )
            self._onehot.fit(categorical_block)

        self.feature_names_out_ = self._build_feature_names()
        self._fitted = True
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor.transform called before fit()")

        blocks: list[np.ndarray] = []

        if self.numeric_columns_:
            numeric_block = self._impute_numeric(X[self.numeric_columns_])
            if self._scaler is not None:
                numeric_block = self._scaler.transform(numeric_block)
            blocks.append(np.asarray(numeric_block, dtype=np.float32))

        if self._missing_indicator_columns:
            indicators = (
                X[self._missing_indicator_columns].isna().to_numpy(dtype=np.float32)
            )
            blocks.append(indicators)

        if self.categorical_columns_:
            categorical_block = self._impute_categorical(X[self.categorical_columns_])
            blocks.append(
                np.asarray(self._onehot.transform(categorical_block), dtype=np.float32)
            )

        if not blocks:
            return np.zeros((len(X), 0), dtype=np.float32)

        matrix = np.hstack(blocks).astype(np.float32, copy=False)
        if matrix.shape[1] != len(self.feature_names_out_):
            raise RuntimeError(
                "feature count mismatch: "
                f"{matrix.shape[1]} produced vs {len(self.feature_names_out_)} named"
            )
        return matrix

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.transform(X), columns=self.feature_names_out_, index=X.index
        )

    # --------------------------------------------------------------- helpers
    def _impute_numeric(self, frame: pd.DataFrame) -> np.ndarray:
        out = frame.copy()
        for column in self.numeric_columns_:
            if out[column].dtype == bool:
                out[column] = out[column].astype(float)
            out[column] = out[column].astype(float).fillna(self._numeric_fill[column])
        return out.to_numpy(dtype=np.float64, copy=False)

    def _impute_categorical(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.categorical_columns_:
            out[column] = (
                out[column]
                .astype("object")
                .where(out[column].notna(), self._categorical_fill[column])
                .astype(str)
            )
        return out

    def _build_feature_names(self) -> list[str]:
        names: list[str] = list(self.numeric_columns_)
        names += [f"{c}__is_missing" for c in self._missing_indicator_columns]
        if self._onehot is not None:
            names += [str(n) for n in self._onehot.get_feature_names_out(
                self.categorical_columns_
            )]
        return names

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "numeric_columns": self.numeric_columns_,
            "categorical_columns": self.categorical_columns_,
            "dropped_columns": self.dropped_columns_,
            "n_features_out": len(self.feature_names_out_),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "TabularPreprocessor":
        return joblib.load(Path(path))
