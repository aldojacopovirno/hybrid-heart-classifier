from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    from imblearn.over_sampling import SMOTE # type: ignore
except Exception:  # fallback if imblearn not available
    SMOTE = None  # type: ignore


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlations across numeric predictors for diagnostics.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numeric columns to evaluate.

    Returns
    -------
    pandas.DataFrame
        Correlation matrix of numeric features.
    """

    return df.corr(numeric_only=True)


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Calculate variance inflation factors for neural network inputs.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset providing feature values.
    features : list of str
        Feature names to test for multicollinearity.

    Returns
    -------
    pandas.DataFrame
        Table of features with associated VIF scores.
    """

    if not features:
        return pd.DataFrame(columns=["feature", "vif"])
    X = df[features].copy()
    X = sm.add_constant(X, has_constant='add')
    vif_data = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'feature': col, 'vif': float(vif) if np.isfinite(vif) else np.nan})
    return pd.DataFrame(vif_data).sort_values('vif', ascending=False).reset_index(drop=True)


@dataclass
class MLPConfig:
    target_col: str = 'target'
    test_size: float = 0.2
    random_state: int = 42
    k_folds: int = 5
    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    activation: str = 'relu'
    alpha: float = 1e-4
    max_iter: int = 1000
    learning_rate: str = 'adaptive'


@dataclass
class MLPArtifacts:
    corr: pd.DataFrame
    vif: pd.DataFrame
    confusion: np.ndarray
    roc_auc_macro: Optional[float]
    fold_scores: List[float]
    fpr: List[np.ndarray]
    tpr: List[np.ndarray]


def split_data(df: pd.DataFrame, cfg: MLPConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide the dataset into stratified training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing predictors and the target label.
    cfg : MLPConfig
        Configuration object describing split proportions and target name.

    Returns
    -------
    tuple of pandas.DataFrame
        Stratified training and testing dataframes.
    """

    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df[cfg.target_col]
    )
    return train_df, test_df


def maybe_smote(X: np.ndarray, y: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Apply SMOTE oversampling when available to balance the classes.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix for the training subset.
    y : numpy.ndarray
        Target labels for the training subset.
    random_state : int
        Random seed used by SMOTE for reproducibility.

    Returns
    -------
    tuple
        Resampled ``X`` and ``y`` arrays plus a dictionary summarizing class
        counts after resampling.
    """

    if SMOTE is None:
        # imblearn not available; return as-is, with counts
        _, counts = np.unique(y, return_counts=True)
        return X, y, {str(i): int(c) for i, c in enumerate(counts)}
    try:
        smt = SMOTE(random_state=random_state)
        X_res, y_res = smt.fit_resample(X, y)
        vals, counts = np.unique(y_res, return_counts=True)
        balance = {int(k): int(v) for k, v in zip(vals, counts)}
        return X_res, y_res, balance
    except Exception:
        # Fallback to original
        vals, counts = np.unique(y, return_counts=True)
        return X, y, {int(k): int(v) for k, v in zip(vals, counts)}


def fit_mlp(train_df: pd.DataFrame, features: List[str], cfg: MLPConfig) -> MLPClassifier:
    """Train a multilayer perceptron classifier with optional SMOTE balancing.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataset containing predictors and target labels.
    features : list of str
        Feature names fed into the neural network.
    cfg : MLPConfig
        Hyperparameters controlling the neural network architecture.

    Returns
    -------
    sklearn.neural_network.MLPClassifier
        Fitted neural network classifier.
    """

    X = train_df[features].values
    y = train_df[cfg.target_col].values
    # Apply SMOTE on training data only
    X_bal, y_bal, balance = maybe_smote(X, y, cfg.random_state)
    fit_mlp.last_balance = balance  # type: ignore[attr-defined]
    clf = MLPClassifier(
        hidden_layer_sizes=cfg.hidden_layer_sizes,
        activation=cfg.activation,
        alpha=cfg.alpha,
        random_state=cfg.random_state,
        max_iter=cfg.max_iter,
        learning_rate=cfg.learning_rate,
    )
    clf.fit(X_bal, y_bal)
    return clf


def predict_proba(clf: MLPClassifier, X: pd.DataFrame) -> np.ndarray:
    """Predict class probabilities using the trained MLP classifier.

    Parameters
    ----------
    clf : sklearn.neural_network.MLPClassifier
        Trained MLP model.
    X : pandas.DataFrame
        Feature matrix for inference.

    Returns
    -------
    numpy.ndarray
        Probability estimates for each class.
    """

    return clf.predict_proba(X.values)


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], cfg: MLPConfig) -> MLPArtifacts:
    """Fit the MLP model, perform diagnostics, and compute evaluation metrics.

    Parameters
    ----------
    train_df : pandas.DataFrame
        Training subset used for model fitting and diagnostics.
    test_df : pandas.DataFrame
        Hold-out subset used for evaluation metrics.
    features : list of str
        Columns supplied to the classifier.
    cfg : MLPConfig
        Configuration with hyperparameters and split details.

    Returns
    -------
    MLPArtifacts
        Artifact bundle containing diagnostics, ROC data, and confusion matrix.
    """

    # Diagnostics
    corr = compute_correlations(train_df[features + [cfg.target_col]])
    vif = compute_vif(train_df, features)

    # Fit model (with SMOTE on train)
    clf = fit_mlp(train_df, features, cfg)
    evaluate_model.last_fit_model = clf  # type: ignore[attr-defined]
    evaluate_model.features = features  # type: ignore[attr-defined]

    # Test evaluation
    X_test = test_df[features]
    y_test = test_df[cfg.target_col].values
    probs = predict_proba(clf, X_test)
    # Map classes
    class_index = {c: i for i, c in enumerate(clf.classes_)}
    y_pred = np.array([np.argmax(probs[i]) for i in range(len(probs))])
    cm = confusion_matrix(y_test, y_pred)

    # ROC per class
    classes = np.sort(np.unique(np.concatenate([train_df[cfg.target_col].values, y_test])))
    fpr_list, tpr_list, roc_auc_list = [], [], []
    for k in classes:
        y_true_bin = (y_test == k).astype(int)
        idx = class_index.get(k, None)
        if idx is None:
            continue
        y_score = probs[:, idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        if len(fpr) and len(tpr):
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(auc(fpr, tpr))
    roc_macro = float(np.mean(roc_auc_list)) if roc_auc_list else None

    # K-fold CV on training set (macro AUC)
    X_all = train_df[features].values
    y_all = train_df[cfg.target_col].values
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_state)
    fold_scores: List[float] = []
    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        # SMOTE on each fold's train
        X_trb, y_trb, _ = maybe_smote(X_tr, y_tr, cfg.random_state)
        clf_cv = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            activation=cfg.activation,
            alpha=cfg.alpha,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
        )
        try:
            clf_cv.fit(X_trb, y_trb)
            probs_va = clf_cv.predict_proba(X_va)
            classes_cv = np.sort(np.unique(y_va))
            aucs = []
            idx_map = {c: i for i, c in enumerate(clf_cv.classes_)}
            for k in classes_cv:
                y_true_bin = (y_va == k).astype(int)
                i_col = idx_map.get(k, None)
                if i_col is None:
                    continue
                fpr, tpr, _ = roc_curve(y_true_bin, probs_va[:, i_col])
                aucs.append(auc(fpr, tpr))
            fold_scores.append(float(np.mean(aucs)) if aucs else np.nan)
        except Exception:
            fold_scores.append(np.nan)

    return MLPArtifacts(
        corr=corr,
        vif=vif,
        confusion=cm,
        roc_auc_macro=roc_macro,
        fold_scores=fold_scores,
        fpr=fpr_list,
        tpr=tpr_list,
    )


def save_artifacts(art: MLPArtifacts, charts_dir: str = 'charts', metrics_dir: str = 'metrics') -> Dict[str, str]:
    """Persist MLP evaluation artifacts (plots, text, CSVs) to disk.

    Parameters
    ----------
    art : MLPArtifacts
        Outputs returned by ``evaluate_model``.
    charts_dir : str, optional
        Directory used for saving figures.
    metrics_dir : str, optional
        Directory used for saving textual summaries and CSV files.

    Returns
    -------
    dict
        Mapping between artifact names and their file paths.
    """

    import os
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    # Correlation heatmap
    if isinstance(art.corr, pd.DataFrame) and not art.corr.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(art.corr, annot=False, cmap='coolwarm', center=0)
        p = f"{charts_dir}/mlp_correlations.png"
        plt.title('Feature Correlations (MLP)')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['corr_heatmap'] = p

    # VIF barplot
    if isinstance(art.vif, pd.DataFrame) and not art.vif.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=art.vif, x='vif', y='feature', orient='h')
        p = f"{charts_dir}/mlp_vif.png"
        plt.title('Variance Inflation Factors (MLP)')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['vif'] = p

    # Confusion matrix heatmap
    if isinstance(art.confusion, np.ndarray) and art.confusion.size:
        plt.figure(figsize=(6, 5))
        sns.heatmap(art.confusion, annot=True, fmt='d', cmap='Blues')
        p = f"{charts_dir}/mlp_confusion.png"
        plt.title('Confusion Matrix (MLP, Test)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['confusion'] = p

    # ROC curves per class (one-vs-rest)
    if art.fpr and art.tpr:
        plt.figure(figsize=(6, 5))
        for i, (fpr, tpr) in enumerate(zip(art.fpr, art.tpr)):
            plt.plot(fpr, tpr, label=f'class {i} ROC')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        title = 'MLP ROC Curves'
        if art.roc_auc_macro is not None:
            title += f' (macro AUC={art.roc_auc_macro:.3f})'
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        p = f"{charts_dir}/mlp_roc.png"
        plt.savefig(p)
        plt.close()
        paths['roc'] = p

    # Save textual metrics
    metrics_path = f"{metrics_dir}/mlp_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write('MLP Metrics and Diagnostics\n')
        f.write('===========================\n')
        f.write(f"Macro ROC AUC (test): {art.roc_auc_macro}\n")
        if len(art.fold_scores):
            f.write(f"CV Macro AUC (mean): {np.nanmean(art.fold_scores):.4f}\n")
            f.write(f"CV Macro AUC (per fold): {art.fold_scores}\n")
        f.write('\nConfusion Matrix (rows=True, cols=Pred):\n')
        f.write(np.array2string(art.confusion))
        if isinstance(art.vif, pd.DataFrame) and not art.vif.empty:
            f.write('\n\n--- VIF ---\n')
            f.write(art.vif.to_string(index=False))
        if isinstance(art.corr, pd.DataFrame) and not art.corr.empty:
            f.write('\n\n--- Correlations (head) ---\n')
            f.write(art.corr.round(3).to_string())
    paths['metrics_txt'] = metrics_path

    # Export CSVs
    if isinstance(art.vif, pd.DataFrame) and not art.vif.empty:
        vif_csv = f"{metrics_dir}/mlp_vif.csv"
        art.vif.to_csv(vif_csv, index=False)
        paths['vif_csv'] = vif_csv
    if isinstance(art.confusion, np.ndarray) and art.confusion.size:
        cm_csv = f"{metrics_dir}/mlp_confusion_matrix.csv"
        pd.DataFrame(art.confusion).to_csv(cm_csv, index=False)
        paths['confusion_csv'] = cm_csv

    return paths


def run_mlp_pipeline(df: pd.DataFrame, target_col: str = 'target') -> Tuple[MLPArtifacts, Dict[str, str]]:
    """Execute the MLP modeling pipeline and persist generated artifacts.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset prepared for modeling.
    target_col : str, optional
        Name of the target column to predict.

    Returns
    -------
    tuple
        MLP artifacts and saved artifact paths.
    """

    cfg = MLPConfig(target_col=target_col)
    features = [c for c in df.columns if c != target_col]
    train_df, test_df = split_data(df, cfg)
    artifacts = evaluate_model(train_df, test_df, features, cfg)
    paths = save_artifacts(artifacts, charts_dir='charts', metrics_dir='metrics')
    return artifacts, paths
