from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


ZERO_IMPUTE_COLS: List[str] = ["ca", "slope", "exang", "fbs", "cp", "thal"]
KNN_IMPUTE_COLS: List[str] = ["trestbps", "chol", "restecg", "thalch", "oldpeak"]


def impute_zeros(df: pd.DataFrame, cols: Iterable[str] = ZERO_IMPUTE_COLS) -> pd.DataFrame:
    """Impute NaNs with 0 only for specified columns.

    Columns not present in df are ignored.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out


def impute_knn(df: pd.DataFrame, cols: Iterable[str] = KNN_IMPUTE_COLS, n_neighbors: int = 5) -> pd.DataFrame:
    """Apply KNNImputer only to specified columns.

    Works with numeric columns; silently skips columns not in df.
    """
    out = df.copy()
    target_cols = [c for c in cols if c in out.columns]
    if not target_cols:
        return out

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(out[target_cols])
    out[target_cols] = imputed
    return out


def robust_scale_all_but_target(df: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    """Apply RobustScaler to all features except target.

    Leaves target untouched if present.
    """
    out = df.copy()
    feature_cols = [c for c in out.columns if c != target_col]
    if not feature_cols:
        return out
    scaler = RobustScaler()
    out[feature_cols] = scaler.fit_transform(out[feature_cols])
    return out


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Full processing pipeline as requested.

    - Zero-impute ca, slope, exang, fbs, cp, thal
    - KNN-impute trestbps, chol, restecg, thalch, oldpeak
    - Robust scale all columns except target
    """
    out = impute_zeros(df)
    out = impute_knn(out)
    out = robust_scale_all_but_target(out, target_col="target")
    return out

