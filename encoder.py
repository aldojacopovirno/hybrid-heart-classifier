from __future__ import annotations

import pandas as pd


def drop_unused_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns not used for modeling/analysis.

    Drops: id, dataset (if present).
    """
    cols_to_drop = [c for c in ["id", "dataset"] if c in df.columns]
    return df.drop(columns=cols_to_drop)


def rename_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rename label column to 'target'. Supports 'num' or typo 'tagrt'."""
    rename_map = {}
    if "num" in df.columns:
        rename_map["num"] = "target"
    if "tagrt" in df.columns:
        rename_map["tagrt"] = "target"
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Map categorical/string fields to numeric codes with explicit dictionaries.

    - sex: Male/Female -> 0/1 (accepts case-insensitive and 0/1 passthrough)
    - cp: typical angina/atypical angina/non-anginal/asymptomatic -> 0/1/2/3
    - fbs: FALSE/TRUE -> 0/1
    - restecg: normal/st-t abnormality/lv hypertrophy -> 0/1/2
    - exang: FALSE/TRUE -> 0/1
    - slope: flat/downsloping/upsloping -> 0/1/2
    - thal: normal/fixed defect/reversible defect -> 0/1/2
    """

    df = df.copy()

    # Normalize helper for strings
    def norm(x):
        if isinstance(x, str):
            return x.strip().lower()
        return x

    # sex mapping: allow numeric already (0/1), or strings
    if "sex" in df.columns:
        sex_map = {"male": 0, "m": 0, 1: 1, "1": 1, "female": 1, "f": 1, 0: 0, "0": 0}
        df["sex"] = df["sex"].map(lambda v: sex_map.get(norm(v), v))

    if "cp" in df.columns:
        cp_map = {
            "typical angina": 0,
            "atypical angina": 1,
            "non-anginal": 2,
            "non anginal": 2,
            "asymptomatic": 3,
            0: 0,
            1: 1,
            2: 2,
            3: 3,
        }
        df["cp"] = df["cp"].map(lambda v: cp_map.get(norm(v), v))

    for col in ["fbs", "exang"]:
        if col in df.columns:
            bin_map = {"false": 0, "true": 1, False: 0, True: 1, 0: 0, 1: 1, "0": 0, "1": 1}
            df[col] = df[col].map(lambda v: bin_map.get(norm(v), v))

    if "restecg" in df.columns:
        restecg_map = {
            "normal": 0,
            "st-t abnormality": 1,
            "st t abnormality": 1,
            "lv hypertrophy": 2,
            0: 0,
            1: 1,
            2: 2,
        }
        df["restecg"] = df["restecg"].map(lambda v: restecg_map.get(norm(v), v))

    if "slope" in df.columns:
        slope_map = {
            "flat": 0,
            "downsloping": 1,
            "down sloping": 1,
            "upsloping": 2,
            "up sloping": 2,
            0: 0,
            1: 1,
            2: 2,
        }
        df["slope"] = df["slope"].map(lambda v: slope_map.get(norm(v), v))

    if "thal" in df.columns:
        thal_map = {
            "normal": 0,
            "fixed defect": 1,
            "reversible defect": 2,
            "reversable defect": 2,
            0: 0,
            1: 1,
            2: 2,
        }
        # Also handle inputs with semicolons or extra spaces
        def norm_thal(v):
            n = norm(v)
            if isinstance(n, str):
                n = n.replace(";", " ").replace("  ", " ")
            return n

        df["thal"] = df["thal"].map(lambda v: thal_map.get(norm_thal(v), v))

    # Enforce integer dtype for encoded columns and target
    int_cols = [c for c in ["age","sex", "trestbps", "chol", "cp", "fbs", "thalch", "restecg", "exang", "slope", "ca", "thal", "target"] if c in df.columns]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in int_cols:
        if not df[c].isna().any():
            df[c] = df[c].astype(int)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Full encoding pipeline.

    Steps:
    - Drop 'id', 'dataset'
    - Rename 'num' -> 'target'
    - Encode categoricals to numeric codes
    """
    df = drop_unused_features(df)
    df = rename_target(df)
    df = encode_categoricals(df)
    return df
