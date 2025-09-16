from __future__ import annotations

import pandas as pd


def encode_thal_to_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``thal`` into mutually exclusive boolean indicator columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset that may contain the integer-encoded ``thal`` feature.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``thal_fixed`` and ``thal_reversible`` indicators and
        the original ``thal`` column removed when present.

    Notes
    -----
    Values of 1 map to ``thal_fixed``, values of 2 map to ``thal_reversible``
    and other entries produce zeros for both indicators.
    """
    out = df.copy()
    if "thal" not in out.columns:
        # Create columns defaulting to 0 if thal missing
        out["thal_fixed"] = 0
        out["thal_reversible"] = 0
        return out

    thal_series = pd.to_numeric(out["thal"], errors="coerce")
    out["thal_fixed"] = (thal_series == 1).astype(int)
    out["thal_reversible"] = (thal_series == 2).astype(int)
    # Entries 0/NaN naturally map to 0 in both
    # Drop original thal column as requested
    out = out.drop(columns=["thal"])  
    return out


def onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply project-specific one-hot encoding transformations.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset with categorical columns requiring bespoke one-hot encoding.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the applied one-hot transformations.
    """
    out = encode_thal_to_booleans(df)
    return out
