# Optimizing Heart Disease Classification Through Statistical-Mathematical Hybridization: A Comparative Analysis

## Description
- Hybrid, modular pipeline for ordinal/multi-class heart disease classification on the UCI dataset.
- End-to-end workflow: encoding, preprocessing, EDA, multiple models (OLR, RF, XGB, MLP), evaluation, and artifact saving.
- Produces charts and metrics with a final comparative report for easy benchmarking.

## Repository Structure
```
.
├── main.py                    # Orchestrates pipeline and model comparison
├── script/
│   ├── encoder.py             # Load, clean, encode categorical/target
│   ├── processing.py          # Imputation and scaling
│   ├── onehot_encoding.py     # Targeted one-hot for thal
│   ├── eda.py                 # EDA and plots
│   ├── olr_model.py           # Ordinal Logistic Regression
│   ├── rf_model.py            # Random Forest
│   ├── xgb_model.py           # XGBoost
│   └── mlp_model.py           # MLP classifier with optional SMOTE
├── data/
│   └── heart_disease_uci.csv  # Input dataset (not tracked publicly)
├── charts/                    # Generated plots (png)
├── metrics/                   # Reports and tables (csv/txt)
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT license
```

## Installation
- Prerequisites: Python 3.10+
- Create environment and install dependencies:
```
python -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
# If needed: pip install imbalanced-learn
```

## Usage
- Place the dataset at `data/heart_disease_uci.csv`.
- Run the full pipeline:
```
python main.py
```
- Outputs:
  - Charts in `charts/`
  - Metrics and reports in `metrics/`
  - Comparative table: `metrics/comparative_metrics.csv`

## Configuration
- Default model settings live in the model modules under `script/`.
- Typical knobs:
  - OLR: distribution, folds, random seed
  - RF: estimators, depth
  - XGB: estimators, depth, learning rate, subsampling
  - MLP: hidden sizes, activation, max_iter, learning rate, SMOTE
- Adjust directly in `script/*.py` or extend `main.py` with CLI args.

## Contributing
- Fork the repo and create a feature branch.
- Keep changes small and focused; follow existing style.
- Include concise descriptions and rationale in PRs.
- Optional: add lightweight tests for data handling and metrics.

## License
- MIT License. See `LICENSE`.

## Authors / Acknowledgments
- Authors: Aldo Jacopo Virno, Andrea Bucchignani
- Data: UCI Heart Disease dataset
- Tooling: scikit-learn, statsmodels, xgboost, pandas, numpy, seaborn, matplotlib
