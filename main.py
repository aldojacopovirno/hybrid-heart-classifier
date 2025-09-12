from __future__ import annotations

import pandas as pd

from encoder import encode


def load_and_encode(csv_path: str = "data/heart_disease_uci.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    encoded_data = encode(df)
    return encoded_data


if __name__ == "__main__":
    encoded_data = load_and_encode()
    print(encoded_data.head()) 