"""Pre-processing utilities for the UCI Heart Disease dataset.

This module provides a reusable `preprocessing` class (lowercase by
explicit UX requirement) that performs:

- loading of a CSV dataset
- light EDA logging (shape, dtypes, nulls, numerics describe, unique values)
- specific transformations (drop, rename, categorical encodings)

Notes
-----
- The class name `preprocessing` intentionally violates common naming
  conventions (PEP8 suggests CapWords) to match a product requirement.
  This deviation is documented here for clarity.
- Transformations are conservative and log warnings for missing columns or
  unexpected values; unknown categorical values map to NaN.

Dependencies
------------
- pandas (numpy is optional and not required here)

Examples
--------
>>> from pre_processing import preprocessing
>>> pp = preprocessing()
>>> df = pp.run()  # uses default path data/heart_disease_uci.csv
>>> df.shape  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _as_path(path: Optional[str | Path]) -> Path:
    return Path(path) if path is not None else Path("data/heart_disease_uci.csv")


def _normalize_string(value: object) -> Optional[str]:
    """Normalize an arbitrary value to a canonical lowercase string.

    Parameters
    ----------
    value : object
        Input value which may be str, number, bool, or None.

    Returns
    -------
    Optional[str]
        Normalized string or None if input is NA-like.
    """
    if pd.isna(value):
        return None
    try:
        s = str(value)
    except Exception:  # pragma: no cover - defensive
        return None
    s = s.strip().lower()
    return s if s != "" else None


def _coerce_numeric_domain(series: pd.Series, valid_values: Iterable[int], col: str) -> pd.Series:
    """Ensure numeric series values lie within a valid domain.

    Invalid values are set to pandas NA with a warning, and dtype coerced to
    nullable integer (Int64) when possible.
    """
    valid_set = set(valid_values)
    s = pd.to_numeric(series, errors="coerce")
    invalid_mask = ~s.isna() & ~s.astype("Int64").isin(valid_set)
    if invalid_mask.any():
        examples = s[invalid_mask].unique()
        LOGGER.warning(
            "Column '%s' contains values outside domain %s; setting to NA. Examples: %s",
            col,
            sorted(valid_set),
            examples[:5],
        )
        s.loc[invalid_mask] = pd.NA
    return s.astype("Int64")


def _map_with_warning(
    series: pd.Series,
    mapping: Mapping[str, int],
    col: str,
    valid_domain: Iterable[int],
) -> pd.Series:
    """Map mixed-type categorical values to integer codes.

    - Applies case-insensitive and trimmed string normalization
    - If the input series is already numeric, validate domain
    - Unknown categorical tokens are mapped to NA with a warning

    Parameters
    ----------
    series : pandas.Series
        Input values (object, bool, str, or numeric).
    mapping : Mapping[str, int]
        Canonical lowercase token -> integer code.
    col : str
        Column name (for logging).
    valid_domain : Iterable[int]
        Set of valid integer values expected for this column.

    Returns
    -------
    pandas.Series
        Nullable integer (Int64) coded series.
    """
    # If numeric, just validate domain and return.
    if pd.api.types.is_numeric_dtype(series):
        return _coerce_numeric_domain(series, valid_domain, col)

    normed = series.map(_normalize_string)
    coded = normed.map(lambda x: mapping.get(x) if x is not None else pd.NA)

    unknown_mask = normed.notna() & coded.isna()
    if unknown_mask.any():
        examples = series[unknown_mask].astype(str).unique()
        LOGGER.warning(
            "Unknown tokens in column '%s' mapped to NA. Examples: %s",
            col,
            examples[:5],
        )
    return coded.astype("Int64")


@dataclass
class preprocessing:
    """Pre-processing pipeline for UCI Heart Disease dataset.

    The class name is intentionally lowercase to satisfy an explicit
    UX requirement (deviates from PEP8 CapWords convention).

    Parameters
    ----------
    input_path : str or pathlib.Path, optional
        CSV file path. Defaults to ``data/heart_disease_uci.csv``.
    logger_level : int, optional
        Python logging level for the module logger. Defaults to ``logging.INFO``.

    Attributes
    ----------
    raw_data : pandas.DataFrame
        The raw dataset loaded from CSV.
    pre_processed_data : pandas.DataFrame
        The processed dataset after transformations.
    """

    input_path: Optional[str | Path] = None
    logger_level: int = logging.INFO
    raw_data: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)
    pre_processed_data: pd.DataFrame = field(init=False, default_factory=pd.DataFrame)

    # Default target mappings and domains
    _SEX_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        # canonical
        "male": 0,
        "m": 0,
        "female": 1,
        "f": 1,
    })
    _CP_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        "typical angina": 0,
        "ta": 0,
        "atypical angina": 1,
        "aa": 1,
        "non-anginal": 2,
        "non anginal": 2,
        "nonanginal": 2,
        "asymptomatic": 3,
    })
    _FBS_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        "false": 0,
        "0": 0,
        "true": 1,
        "1": 1,
    })
    _RESTECG_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        "normal": 0,
        "st-t abnormality": 1,
        "st t abnormality": 1,
        "st_t abnormality": 1,
        "lv hypertrophy": 2,
        "left ventricular hypertrophy": 2,
    })
    _EXANG_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        "false": 0,
        "0": 0,
        "true": 1,
        "1": 1,
    })
    _SLOPE_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        "flat": 0,
        "downsloping": 1,
        "down sloping": 1,
        "down": 1,
        "upsloping": 2,
        "up sloping": 2,
        "up": 2,
    })
    _THAL_MAP: Dict[str, int] = field(init=False, default_factory=lambda: {
        "normal": 0,
        "fixed defect": 1,
        "fixed": 1,
        "reversible defect": 2,
        "reversible": 2,
    })

    def __post_init__(self) -> None:
        logging.basicConfig(level=self.logger_level, format="%(levelname)s: %(message)s")
        self.input_path = _as_path(self.input_path)

    # ---- Public API -----------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """Load CSV into a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Loaded raw dataset.

        Raises
        ------
        FileNotFoundError
            If the CSV path does not exist.
        ValueError
            If the file cannot be parsed as CSV.
        """
        path = self.input_path
        assert path is not None  # for typing

        if not Path(path).exists():
            LOGGER.error("Input file not found: %s", path)
            raise FileNotFoundError(f"Input file not found: {path}")

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - pandas parser variations
            LOGGER.error("Failed to read CSV: %s", exc)
            raise ValueError(f"Failed to read CSV: {exc}") from exc

        # Basic whitespace trim for string/object columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()

        dup_count = int(df.duplicated().sum())
        if dup_count:
            LOGGER.warning("Found %d duplicated rows (not dropped).", dup_count)
        else:
            LOGGER.info("No duplicated rows found.")

        self.raw_data = df
        return df

    def explore_data(self, df: Optional[pd.DataFrame] = None) -> None:
        """Perform a concise EDA and log results.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame to explore. Defaults to ``self.raw_data`` when omitted.
        """
        data = df if df is not None else self.raw_data
        if data.empty:
            LOGGER.warning("EDA skipped: empty DataFrame.")
            return

        LOGGER.info("EDA: shape = %s", data.shape)
        LOGGER.info("EDA: dtypes =\n%s", data.dtypes)
        LOGGER.info("EDA: nulls per column =\n%s", data.isnull().sum())

        num = data.select_dtypes(include=["number"])  # numeric statistics
        if not num.empty:
            LOGGER.info("EDA: numeric describe =\n%s", num.describe())
        else:
            LOGGER.info("EDA: no numeric columns detected.")

        # Unique values for selected categoricals
        cat_cols = [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "thal",
        ]
        for c in cat_cols:
            if c in data.columns:
                uniq = data[c].dropna().unique()
                LOGGER.info("EDA: unique values for '%s' (n=%d): %s", c, len(uniq), uniq[:15])
            else:
                LOGGER.warning("EDA: column '%s' not found.", c)

        LOGGER.info("EDA: head =\n%s", data.head())

    def drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop optional features if present.

        Drops columns ``id`` and ``dataset`` when present.
        """
        to_drop = [c for c in ["id", "dataset"] if c in df.columns]
        if to_drop:
            LOGGER.info("Dropping columns: %s", to_drop)
            return df.drop(columns=to_drop)
        LOGGER.info("No optional columns to drop.")
        return df

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename ``num`` to ``target`` with priority rules.

        - If ``target`` exists already, it takes priority and ``num`` is kept as-is
          with a warning (no silent overwrite).
        - If only ``num`` exists, it is renamed to ``target``.
        - If neither exists, no action is taken.
        """
        has_num = "num" in df.columns
        has_target = "target" in df.columns
        if has_num and not has_target:
            LOGGER.info("Renaming 'num' -> 'target'.")
            return df.rename(columns={"num": "target"})
        if has_num and has_target:
            LOGGER.warning(
                "Both 'num' and 'target' present; keeping 'target' (priority) and leaving 'num' unchanged."
            )
            return df
        if has_target:
            LOGGER.info("'target' column already present; no rename performed.")
        else:
            LOGGER.warning("Neither 'num' nor 'target' found; no target column present.")
        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features to integers with specified mappings.

        Features encoded (if present): ``sex``, ``cp``, ``fbs``, ``restecg``,
        ``exang``, ``slope``, ``thal``.

        Unknown tokens are mapped to NA and a warning is logged.
        """
        encoded = df.copy()

        # sex: Male -> 0, Female -> 1
        if "sex" in encoded.columns:
            encoded["sex"] = _map_with_warning(
                encoded["sex"], self._SEX_MAP, "sex", valid_domain=[0, 1]
            )
        else:
            LOGGER.warning("Column 'sex' not found for encoding.")

        # cp: typical angina -> 0, atypical angina -> 1, non-anginal -> 2, asymptomatic -> 3
        if "cp" in encoded.columns:
            encoded["cp"] = _map_with_warning(
                encoded["cp"], self._CP_MAP, "cp", valid_domain=[0, 1, 2, 3]
            )
        else:
            LOGGER.warning("Column 'cp' not found for encoding.")

        # fbs: FALSE -> 0, TRUE -> 1
        if "fbs" in encoded.columns:
            encoded["fbs"] = _map_with_warning(
                encoded["fbs"], self._FBS_MAP, "fbs", valid_domain=[0, 1]
            )
        else:
            LOGGER.warning("Column 'fbs' not found for encoding.")

        # restecg: normal -> 0, st-t abnormality -> 1, lv hypertrophy -> 2
        if "restecg" in encoded.columns:
            encoded["restecg"] = _map_with_warning(
                encoded["restecg"], self._RESTECG_MAP, "restecg", valid_domain=[0, 1, 2]
            )
        else:
            LOGGER.warning("Column 'restecg' not found for encoding.")

        # exang: FALSE -> 0, TRUE -> 1
        if "exang" in encoded.columns:
            encoded["exang"] = _map_with_warning(
                encoded["exang"], self._EXANG_MAP, "exang", valid_domain=[0, 1]
            )
        else:
            LOGGER.warning("Column 'exang' not found for encoding.")

        # slope: flat -> 0, downsloping -> 1, upsloping -> 2
        if "slope" in encoded.columns:
            encoded["slope"] = _map_with_warning(
                encoded["slope"], self._SLOPE_MAP, "slope", valid_domain=[0, 1, 2]
            )
        else:
            LOGGER.warning("Column 'slope' not found for encoding.")

        # thal: normal -> 0, fixed defect -> 1, reversible defect -> 2
        if "thal" in encoded.columns:
            encoded["thal"] = _map_with_warning(
                encoded["thal"], self._THAL_MAP, "thal", valid_domain=[0, 1, 2]
            )
        else:
            LOGGER.warning("Column 'thal' not found for encoding.")

        return encoded

    def run(self) -> pd.DataFrame:
        """Run the full pre-processing pipeline.

        Returns
        -------
        pandas.DataFrame
            The processed dataset assigned also to ``pre_processed_data``.
        """
        df = self.load_data()
        self.explore_data(df)
        df = self.drop_features(df)
        df = self.rename_columns(df)
        df = self.encode_categoricals(df)
        self.pre_processed_data = df

        # Log final summary
        LOGGER.info("Processed shape = %s", df.shape)
        LOGGER.info("Processed columns = %s", list(df.columns)[:15])
        return df

