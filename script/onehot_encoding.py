from __future__ import annotations

import pandas as pd


def encode_thal_to_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean features thal_fixed and thal_reversible from 'thal'.

    Rules:
    - thal == 1 -> thal_fixed = 1, thal_reversible = 0
    - thal == 2 -> thal_fixed = 0, thal_reversible = 1
    - thal == 0 or other/NaN -> thal_fixed = 0, thal_reversible = 0
    Drops the original 'thal' column after creating the booleans.
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
    """Wrapper to apply one-hot transformations required by the project."""
    out = encode_thal_to_booleans(df)
    return out
