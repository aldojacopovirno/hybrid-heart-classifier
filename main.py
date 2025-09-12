from __future__ import annotations

import pandas as pd

from encoder import encode
from processing import process
from onehot_encoding import onehot_encode
from eda import run_full_eda
from olr_model import run_olr_pipeline
from rf_model import run_rf_pipeline


def load_and_encode(csv_path: str = "data/heart_disease_uci.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    encoded_data = encode(df)
    return encoded_data

if __name__ == "__main__":
    encoded_data = load_and_encode()
    # Process dataset (imputations + scaling)
    processed_data = process(encoded_data)
    # Apply one-hot encoding for 'thal' into boolean features (model input)
    onehot_data = onehot_encode(processed_data)

    print("Encoded data preview:")
    print(encoded_data.head())

    print("\nProcessed data preview:")
    print(processed_data.head())

    print("\nOne-hot data preview:")
    print(onehot_data.head())

    print("\nRunning EDA...")
    # Use processed dataset for EDA
    eda_results = run_full_eda(processed_data, charts_dir="charts")

    print("\nSummary statistics:")
    print(eda_results["summary"])  # DataFrame

    print("\nMissing values per column:")
    print(eda_results["missing"])  # Series

    print("\nOutliers (IQR rule):")
    print(eda_results["outliers"])  # DataFrame

    print("\nCharts saved:")
    for feat, path in eda_results["charts"].items():
        print(f"- {feat}: {path}")

    # Run OLR pipeline on one-hot dataset and save performance
    print("\nTraining OLR model and evaluating...")
    _, saved = run_olr_pipeline(onehot_data, target_col="target")
    print("Saved OLR artifacts (charts + metrics):")
    for k, v in saved.items():
        print(f"- {k}: {v}")

    # Run Random Forest pipeline on one-hot dataset and save performance
    print("\nTraining Random Forest model and evaluating...")
    _, rf_saved = run_rf_pipeline(onehot_data, target_col="target")
    print("Saved RF artifacts (charts + metrics):")
    for k, v in rf_saved.items():
        print(f"- {k}: {v}")
