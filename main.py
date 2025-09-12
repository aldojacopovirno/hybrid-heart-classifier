"""CLI entrypoint for running the UCI Heart Disease pre-processing pipeline.

This module wires the reusable `preprocessing` class from `pre_processing`
and exposes a small command-line interface for convenience.

Examples
--------
Run with defaults (saves to data/pre_processed_heart_disease_uci.csv):
    python main.py

Run with explicit input and output paths:
    python main.py --input-path data/heart_disease_uci.csv --output-path data/custom.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pre_processing import preprocessing


DEFAULT_INPUT = "data/heart_disease_uci.csv"
DEFAULT_OUTPUT = "data/pre_processed_heart_disease_uci.csv"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EDA and pre-processing for the UCI Heart Disease dataset. "
            "Logs EDA via stdlib logging; prints a brief final summary."
        )
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT,
        help=(
            "Output CSV path to persist pre-processed data. "
            f"Default: {DEFAULT_OUTPUT}"
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None):
    """Run the pre-processing pipeline from the command line.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments for testing. Defaults to ``None`` which consumes sys.argv.

    Returns
    -------
    pandas.DataFrame
        The processed dataset as ``pre_processed_data``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    pp = preprocessing(input_path=args.input_path)
    pre_processed_data = pp.run()

    # Simple stdout summary for the user
    encoded_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
    present_encoded = [c for c in encoded_cols if c in pre_processed_data.columns]
    print("=== Pre-processing summary ===")
    print(f"Input: {args.input_path}")
    print(f"Rows, Columns: {pre_processed_data.shape}")
    print(f"Has target: {'target' in pre_processed_data.columns}")
    if present_encoded:
        dtypes_info = {c: str(pre_processed_data[c].dtype) for c in present_encoded}
        print("Encoded columns and dtypes:", dtypes_info)
    else:
        print("No expected categorical columns found to encode.")

    # Persist to disk (default to data/ path if not overridden)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pre_processed_data.to_csv(out_path, index=False)
    print(f"Saved processed CSV to: {out_path}")

    return pre_processed_data


if __name__ == "__main__":
    main()
