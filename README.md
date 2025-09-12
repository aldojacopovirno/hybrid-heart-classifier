# Optimizing Heart Disease Classification Through Statistical-Mathematical Hybridization: A Comparative Analysis

A hybrid pipeline for multi-class/ordinal classification of heart disease severity on tabular data (UCI). The project implements a complete, modular workflow for data preparation, EDA, modeling with multiple algorithms, evaluation, and artifact traceability, following sound software engineering practices.

## Abstract
- Goal: build and evaluate an end-to-end pipeline for heart disease classification (UCI Heart Disease), including data preparation, exploratory analysis, multi-model training, and final comparison.
- Contributions: modular data→model chain (Ordinal Logistic Regression, Random Forest, XGBoost, MLP), robust metrics (macro AUC, MCC, balanced accuracy), diagnostics (VIF, correlations, Brant proxy, Box–Tidwell), and reproducible artifact saving.
- Outputs: plots in `charts/`, reports and metric tables in `metrics/`, and a model comparison in `metrics/comparative_metrics.csv`.

## Dataset
- Source: UCI Heart Disease (Cleveland, Statlog) consolidated into a single CSV.
- Expected path: `data/heart_disease_uci.csv`.
- Target: column `num` renamed to `target` with ordinal values 0–4 (absence → increasing severity).
- Loading and initial encoding: `main.py` uses `load_and_encode()` to read the CSV and invoke the encoder.
- Column names: the pipeline assumes standard names (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach/thalch`, `exang`, `oldpeak`, `slope`, `ca`, `thal`). Some modules reference `thalch` (without the second “a”); ensure the CSV matches or update the code accordingly.

## Architecture
- Ingest & encoding
  - `encoder.py`: drops unused fields (`id`, `dataset`), renames `num`→`target`, and applies explicit mappings for categoricals (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`).
- Preprocessing
  - `processing.py`: zero-imputation for discrete vars (`ca`, `slope`, `exang`, `fbs`, `cp`, `thal`); KNNImputer for continuous (`trestbps`, `chol`, `restecg`, `thalch`, `oldpeak`); RobustScaler on all features except `target`.
- Targeted one-hot
  - `onehot_encoding.py`: maps `thal` to two binary features `thal_fixed`, `thal_reversible` and drops `thal`.
- EDA
  - `eda.py`: extended descriptive stats, missing count, IQR-based outlier detection, and distribution plots (hist+KDE) with normal overlays; saves one PNG per feature to `charts/`.
- Modeling
  - OLR: `olr_model.py` (statsmodels OrderedModel) with diagnostics (correlations, VIF, Box–Tidwell for continuous variables, Brant proxy), stratified split, one-vs-rest ROC macro AUC, and k-fold CV.
  - RF: `rf_model.py` (RandomForestClassifier) with feature importance, ROC OvR macro AUC, and CV.
  - XGB: `xgb_model.py` (XGBClassifier, `multi:softprob`) with ROC OvR macro AUC and CV.
  - MLP: `mlp_model.py` (MLPClassifier) with optional SMOTE on training, ROC OvR macro AUC, and CV.
- Model comparison
  - `main.py`: builds a comparative table (accuracy, macro/weighted precision/recall/F1, MCC, balanced accuracy, macro specificity) saved as `metrics/comparative_metrics.csv`.

## Getting Started
### Prerequisites
- Python ≥ 3.10 (Linux/Mac/Windows).

### Installation
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
# If missing: pip install imbalanced-learn
```
Note: the correct package for SMOTE is `imbalanced-learn`. If it is not included in `requirements.txt`, install it as shown above.

### Run
```bash
python main.py
```
- Ensure `data/heart_disease_uci.csv` exists at the expected path.
- Outputs: dataframe previews in stdout; PNGs in `charts/`; reports and CSVs in `metrics/`; final comparison in `metrics/comparative_metrics.csv`.

## Project Structure
- `main.py`: end-to-end orchestration, artifact saving, and model comparison.
- `encoder.py`: drop/rename/categorical encoding (robust to text/0–1 variants).
- `processing.py`: imputations (zero and KNN) and robust scaling.
- `onehot_encoding.py`: targeted one-hot for `thal`.
- `eda.py`: full EDA and per-feature plots.
- `olr_model.py`, `rf_model.py`, `xgb_model.py`, `mlp_model.py`: training, evaluation, and artifact saving.
- `charts/`: generated plots (PNGs).
- `metrics/`: textual reports and tables (CSV/TXT).
- `requirements.txt`: runtime dependencies.

## Pipeline Details
- Initial encoding: explicit, case-insensitive mappings; `target` coerced to integer when possible.
- Preprocessing: zero-imputation for discrete, KNNImputer for continuous, RobustScaler to mitigate outliers.
- EDA: `feature_summary()` computes count, missing, mean/median, std/var, quartiles, IQR, skewness, kurtosis; IQR-rule for outliers.
- Models and metrics:
  - Stratified split (`test_size=0.2`, `random_state=42`).
  - Test macro AUC and 5-fold CV macro AUC.
  - Confusion matrix; accuracy; macro/weighted precision, recall, F1; MCC; balanced accuracy; macro specificity (derived in `main.py`).
  - OLR: log-transform on all non-target features to stabilize scales; diagnostics VIF, Box–Tidwell (continuous with >10 unique values), and Brant proxy.
  - RF/XGB: feature importance; VIF and correlation diagnostics on training.
  - MLP: SMOTE on training; CV with SMOTE per fold.

## Generated Artifacts
- EDA: `charts/<col>_distribution.png` and summaries in `metrics/`.
- OLR: `charts/olr_*.png`; `metrics/olr_metrics.txt`, `metrics/vif.csv`, `metrics/box_tidwell.csv`, `metrics/brant_proxy.csv`, `metrics/confusion_matrix.csv`.
- RF: `charts/rf_*.png`; `metrics/rf_metrics.txt`, `metrics/rf_vif.csv`, `metrics/rf_feature_importance.csv`, `metrics/rf_confusion_matrix.csv`.
- XGB: `charts/xgb_*.png`; `metrics/xgb_metrics.txt`, `metrics/xgb_vif.csv`, `metrics/xgb_feature_importance.csv`, `metrics/xgb_confusion_matrix.csv`.
- MLP: `charts/mlp_*.png`; `metrics/mlp_metrics.txt`, `metrics/mlp_vif.csv`, `metrics/mlp_confusion_matrix.csv`.
- Comparison: `metrics/comparative_metrics.csv`.

## Configuration
- Model parameters via dataclasses:
  - OLR (`olr_model.py`): `distr='logit'`, `k_folds=5`, `random_state=42`.
  - RF (`rf_model.py`): `n_estimators=500`, `max_depth=None`, `class_weight=None`.
  - XGB (`xgb_model.py`): `n_estimators=300`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.9`.
  - MLP (`mlp_model.py`): `hidden_layer_sizes=(64,32)`, `activation='relu'`, `max_iter=1000`, `learning_rate='adaptive'`.
- To modify: update the dataclasses or create variant pipelines.

## Reproducibility
- Fixed seeds (`random_state=42`) for splits and models where applicable.
- SMOTE and KNNImputer may introduce variability; pin dependency versions for stable results.
- Consider `requirements-lock.txt` (pinned versions) or a Conda environment.

## Quality and Maintainability
- Modularity: dedicated modules per stage with `pandas.DataFrame` interfaces.
- Lightweight typing (type hints) and docstrings across key modules.
- Logging: currently stdout; use configurable `logging` for production.
- Tests: none yet; suggested unit tests for encoding, imputations, splits, and MCC.
- CLI: can extend `main.py` with `argparse` for I/O paths and model selection.

## Known Limitations
- `requirements.txt` previously listed `imblearn.over_sampling` (not installable); use `imbalanced-learn`.
- `thalach` vs `thalch` column name: ensure consistency in the CSV or code (`processing.py`, `encoder.py`).
- Brant test: implemented as a heuristic proxy due to lack of a native function in statsmodels.
- `onehot_encoding.py` drops `thal` after encoding; verify implications for OLR/EDA.

## Future Work
- Hyperparameter search (Grid/Random/Bayesian) with nested CV.
- Sklearn composite pipelines with `ColumnTransformer` and full categorical handling.
- Experiment tracking via MLflow or DVC; data versioning.
- Package as an installable module with a CLI entrypoint.
- Containerization (Docker) for reproducible environments.

## License
This project is released under the MIT License. See `LICENSE` for details. Ensure your use of the UCI dataset complies with its terms.

## Authors
- Aldo Jacopo Virno
- Andrea Bucchignani

## References
- UCI Heart Disease Dataset — UCI ML Repository.
- McCullagh, P. (1980). Regression models for ordinal data. JRSS B.
- Breiman, L. (2001). Random forests. Machine Learning.
- Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Chawla, N. V., et al. (2002). SMOTE. JAIR.
- Software: scikit-learn, statsmodels, xgboost, seaborn, matplotlib, pandas, numpy.

