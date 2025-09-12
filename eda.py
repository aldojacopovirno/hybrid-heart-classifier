from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


def ensure_charts_dir(path: str = "charts") -> str:
    os.makedirs(path, exist_ok=True)
    return path


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive stats for each numeric feature.

    Includes: count, missing, mean, median, std, var, min, q1, q3, max,
    skewness, kurtosis (Fisher), iqr.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    desc = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
    # Additional statistics
    skew = numeric_df.skew()
    kurt = numeric_df.kurtosis()  # Fisher definition (excess)
    missing = df.shape[0] - numeric_df.count()
    var = numeric_df.var()
    iqr = desc["75%"] - desc["25%"]

    out = desc.rename(columns={
        "50%": "median",
        "25%": "q1",
        "75%": "q3",
    })
    out["missing"] = missing
    out["var"] = var
    out["iqr"] = iqr
    out["skewness"] = skew
    out["kurtosis"] = kurt
    # Order columns
    cols = [
        "count", "missing", "mean", "median", "q1", "q3",
        "std", "var", "min", "max", "iqr", "skewness", "kurtosis",
    ]
    return out.reindex(columns=cols)


def detect_missing(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column."""
    return df.isna().sum()


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Flag outliers using IQR rule for numeric columns.

    Returns a DataFrame with columns: lower_bound, upper_bound, outlier_count, outlier_ratio.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask = (numeric_df.lt(lower)) | (numeric_df.gt(upper))
    out_counts = outlier_mask.sum()
    ratios = out_counts / len(numeric_df) if len(numeric_df) else 0
    res = pd.DataFrame({
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": out_counts,
        "outlier_ratio": ratios,
    })
    return res


def plot_feature_distributions(
    df: pd.DataFrame,
    charts_dir: str = "charts",
    figsize: Tuple[int, int] = (8, 5),
) -> Dict[str, str]:
    """Plot histogram + KDE for each numeric feature with overlaid normal PDF.

    Saves one PNG per feature into charts_dir. Returns a mapping feature->filepath.
    """
    ensure_charts_dir(charts_dir)
    sns.set(style="whitegrid")
    paths: Dict[str, str] = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        mean = series.mean()
        std = series.std(ddof=1)

        fig, ax = plt.subplots(figsize=figsize)
        # Histogram and KDE
        sns.histplot(series, bins=30, stat="density", kde=True, color="#4C78A8", alpha=0.6, ax=ax)

        # Normal overlay using statsmodels for PDF generation (via scipy under the hood)
        x = np.linspace(series.min(), series.max(), 200)
        if std and std > 0:
            pdf = stats.norm.pdf(x, loc=mean, scale=std)
            ax.plot(x, pdf, color="#F58518", lw=2, label=f"Normal(mean={mean:.2f}, sd={std:.2f})")

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = os.path.join(charts_dir, f"{col}_distribution.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        paths[col] = out_path

    return paths


def run_full_eda(df: pd.DataFrame, charts_dir: str = "charts") -> Dict[str, Any]:
    """Run full EDA suite and return results in a dictionary.

    Returns keys: summary (DataFrame), missing (Series), outliers (DataFrame), charts (dict)
    """
    results: Dict[str, Any] = {}
    results["summary"] = feature_summary(df)
    results["missing"] = detect_missing(df)
    results["outliers"] = detect_outliers_iqr(df)
    results["charts"] = plot_feature_distributions(df, charts_dir=charts_dir)
    return results