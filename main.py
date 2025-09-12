from __future__ import annotations

import pandas as pd

from encoder import encode
from processing import process
from onehot_encoding import onehot_encode
from eda import run_full_eda


def load_and_encode(csv_path: str = "data/heart_disease_uci.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    encoded_data = encode(df)
    return encoded_data

if __name__ == "__main__":
    encoded_data = load_and_encode()
    # Process dataset (imputations + scaling)
    processed_data = process(encoded_data)
    # Apply one-hot encoding for 'thal' into boolean features
    onehot_data = onehot_encode(processed_data)

    print("Encoded data preview:")
    print(encoded_data.head())

    print("\nProcessed data preview:")
    print(processed_data.head())

    print("\nOne-hot data preview:")
    print(onehot_data.head())

    print("\nRunning EDA...")
    # Use processed dataset for analyses
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
