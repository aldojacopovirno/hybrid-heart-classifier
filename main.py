from __future__ import annotations

import pandas as pd

from pre_processing import preprocess


def load_and_preprocess(csv_path: str = "data/heart_disease_uci.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    pre_processed_data = preprocess(df)
    return pre_processed_data


if __name__ == "__main__":
    pre_processed_data = load_and_preprocess()
    # Simple preview
    print(pre_processed_data.head()) 