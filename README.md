# Optimizing Heart Disease Classification Through Statistical–Mathematical Hybridization

A modular research pipeline for ordinal and multi-class heart disease classification on the UCI Cleveland dataset. The project compares probabilistic (Ordinal Logistic Regression), tree-based (Random Forest, XGBoost), and neural (MLP) baselines deployed under a shared preprocessing, evaluation, and reporting framework.

```
├── main.py                     # Entry point: orchestrates the full pipeline
├── script/
│   ├── encoder.py              # Data ingestion, categorical encoding
│   ├── processing.py           # Imputation and scaling routines
│   ├── onehot_encoding.py      # Specialized one-hot encoding for 'thal'
│   ├── eda.py                  # EDA utilities and plotting
│   ├── olr_model.py            # Ordinal Logistic Regression implementation
│   ├── rf_model.py             # Random Forest wrapper with evaluation hooks
│   ├── xgb_model.py            # XGBoost model wrapper
│   └── mlp_model.py            # MLP classifier and SMOTE integration
├── data/
│   └── heart_disease_uci.csv   # Raw dataset (ignored in VCS)
├── charts/                     # Generated figures
├── metrics/                    # Metrics tables and textual reports
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

## 1. Dataset
- **Source**: UCI Heart Disease (Cleveland subset).
- **Observations**: 303 patients, 6 categorical, 7 numerical predictors.
- **Target**: `num` (0: no disease, 1–4: increasing severity).
- **Preprocessing**:
  - Handle missing values (e.g., `ca`, `thal`) via median/most-frequent imputation.
  - Ordinal-encode target, one-hot encode `thal`, integer encode other categoricals.
  - Standardize continuous variables with `StandardScaler`.

> Place the raw CSV at `data/heart_disease_uci.csv`.

## 2. Pipeline Architecture (`main.py`)
1. **Load & Encode** (`script/encoder.py`, `script/onehot_encoding.py`): data ingestion, categorical transforms.
2. **Processing** (`script/processing.py`): type casting, imputation, scaling.
3. **EDA** (`script/eda.py`): summary statistics, correlation heatmaps, density plots, class distribution charts (saved under `charts/`).
4. **Model Zoo**:
   - `script/olr_model.py`: Ordinal Logistic Regression using `statsmodels`.
   - `script/rf_model.py`: RandomForestClassifier with feature importance export.
   - `script/xgb_model.py`: Gradient boosted trees with early stopping, supports GPU if `tree_method='gpu_hist'`.
   - `script/mlp_model.py`: Multilayer Perceptron with optional SMOTE oversampling for minority classes.
5. **Evaluation** (`main.py` orchestrator):
   - Stratified train/test split (default 80/20, configurable).
   - Cross-validation when supported (OLR with `statsmodels` built-in CV).
   - Metrics: accuracy, F1 (macro & weighted), balanced accuracy, ordinal-aware Cohen’s kappa.
   - Confusion matrices saved per model.
6. **Reporting**:
   - Consolidated metrics table (`metrics/comparative_metrics.csv`).
   - Model-specific reports (`metrics/{model_name}_metrics.txt`).
   - Serialized artifacts if enabled (see configuration).

## 3. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
# Optional extras:
# pip install imbalanced-learn xgboost==1.7.*
```
- Python 3.10+ recommended.
- Ensure `gcc`/`clang` toolchain for compiling some `xgboost` wheels on Linux.

## 4. Running Experiments
```bash
python main.py \
  --test-size 0.2 \
  --random-state 42 \
  --use-smote True \
  --export-models True
```

### Outputs
- Plots: `charts/` (`eda_*`, `confusion_matrix_*`).
- Metrics: `metrics/` (`*_metrics.txt`, `comparative_metrics.csv`).
- Optional serialized models: `artifacts/` (create automatically when `--export-models` is set).

## 5. Configuration Overview
| Component | Key Parameters | Location |
|-----------|----------------|----------|
| Train/Test Split | `test_size`, `random_state` | `main.py` CLI |
| OLR | `method`, `maxiter`, `disp` | `script/olr_model.py` |
| RF | `n_estimators`, `max_depth`, `class_weight` | `script/rf_model.py` |
| XGB | `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `eval_metric` | `script/xgb_model.py` |
| MLP | `hidden_layer_sizes`, `activation`, `learning_rate_init`, `max_iter`, `use_smote` | `script/mlp_model.py` |

Adjust defaults either via CLI flags (extend `argparse` block in `main.py`) or by editing the respective config dictionaries.

## 6. Development Notes
- Follow PEP 8, use type hints where practical.
- Prefer `poetry run`/`pip-tools` if migrating dependency management; update instructions accordingly.
- Run linting (`flake8`, `black --check`) before committing (configure via `pre-commit` if desired).
- Unit tests: add `pytest` suites for data transformers and metric computations; mocks for file IO recommended.

## 7. Contributing
1. Fork and branch from `main`.
2. Document experimental settings in PR descriptions (split seed, hyperparameters).
3. Attach key plots or metrics as PR artifacts.
4. Ensure generated assets are reproducible; do not commit large binaries.

## 8. License
Released under the MIT License - see `LICENSE`.