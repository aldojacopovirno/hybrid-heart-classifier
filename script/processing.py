from __future__ import annotations

from typing import Iterable, List

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler


ZERO_IMPUTE_COLS: List[str] = ["ca", "slope", "exang", "fbs", "cp", "thal"]
KNN_IMPUTE_COLS: List[str] = ["trestbps", "chol", "restecg", "thalch", "oldpeak"]


def impute_zeros(df: pd.DataFrame, cols: Iterable[str] = ZERO_IMPUTE_COLS) -> pd.DataFrame:
    """Fill missing values with zero for selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing potential missing entries.
    cols : Iterable of str, optional
        Column names where zeros should replace missing values.

    Returns
    -------
    pandas.DataFrame
        DataFrame with specified columns imputed while others remain untouched.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out


def impute_knn(df: pd.DataFrame, cols: Iterable[str] = KNN_IMPUTE_COLS, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing numeric features using a K-nearest neighbors strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numeric columns requiring imputation.
    cols : Iterable of str, optional
        Columns eligible for the KNN-based imputations.
    n_neighbors : int, optional
        Number of neighbors used in the imputation algorithm.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the selected columns imputed via ``KNNImputer``.
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
    """Scale feature columns using ``RobustScaler`` while preserving the target.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing both features and the target column.
    target_col : str, optional
        Name of the target column that should remain unchanged.

    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled features and the target column untouched.
    """
    out = df.copy()
    feature_cols = [c for c in out.columns if c != target_col]
    if not feature_cols:
        return out
    scaler = RobustScaler()
    out[feature_cols] = scaler.fit_transform(out[feature_cols])
    return out


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the default preprocessing pipeline for modeling.

    Parameters
    ----------
    df : pandas.DataFrame
        Encoded dataset awaiting imputation and scaling.

    Returns
    -------
    pandas.DataFrame
        Processed dataset with zero-imputed categorical columns, KNN-imputed
        continuous columns, and robust scaling applied to features.
    """
    out = impute_zeros(df)
    out = impute_knn(out)
    out = robust_scale_all_but_target(out, target_col="target")
    return out
