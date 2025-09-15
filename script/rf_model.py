from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# -------------------------
# Data diagnostics helpers
# -------------------------

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame(columns=["feature", "vif"])
    X = df[features].copy()
    # Add constant to stabilize VIF computation
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


# -------------------------
# Modeling helpers
# -------------------------

@dataclass
class RFConfig:
    target_col: str = 'target'
    test_size: float = 0.2
    random_state: int = 42
    k_folds: int = 5
    n_estimators: int = 500
    max_depth: Optional[int] = None
    class_weight: Optional[str] = None  # e.g., 'balanced'


@dataclass
class RFArtifacts:
    corr: pd.DataFrame
    vif: pd.DataFrame
    confusion: np.ndarray
    roc_auc_macro: Optional[float]
    fold_scores: List[float]
    fpr: List[np.ndarray]
    tpr: List[np.ndarray]
    feature_importances_: Optional[pd.DataFrame]


def split_data(df: pd.DataFrame, cfg: RFConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df[cfg.target_col]
    )
    return train_df, test_df


def fit_rf(train_df: pd.DataFrame, features: List[str], cfg: RFConfig) -> RandomForestClassifier:
    X = train_df[features].values
    y = train_df[cfg.target_col].values
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        random_state=cfg.random_state,
        n_jobs=-1,
        class_weight=cfg.class_weight,
    )
    clf.fit(X, y)
    return clf


def predict_proba(clf: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(X.values)


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], cfg: RFConfig) -> RFArtifacts:
    # Diagnostics on training data
    corr = compute_correlations(train_df[features + [cfg.target_col]])
    vif = compute_vif(train_df, features)

    # Fit model
    clf = fit_rf(train_df, features, cfg)
    evaluate_model.last_fit_model = clf  # type: ignore[attr-defined]
    evaluate_model.features = features  # type: ignore[attr-defined]

    # Confusion matrix on test
    X_test = test_df[features]
    y_test = test_df[cfg.target_col].values
    probs = predict_proba(clf, X_test)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # ROC (OvR) per class + macro average AUC
    classes = np.sort(np.unique(np.concatenate([train_df[cfg.target_col].values, y_test])))
    fpr_list, tpr_list, roc_auc_list = [], [], []
    # Align columns: sklearn predict_proba columns follow clf.classes_
    class_index = {c: i for i, c in enumerate(clf.classes_)}
    for k in classes:
        y_true_bin = (y_test == k).astype(int)
        idx = class_index.get(k, None)
        if idx is None:
            # if a class is not present in test predictions, skip
            continue
        y_score = probs[:, idx]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        if len(fpr) and len(tpr):
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(auc(fpr, tpr))
    roc_macro = float(np.mean(roc_auc_list)) if roc_auc_list else None

    # K-fold CV on training set using macro-averaged AUC (OvR)
    # Use a custom scorer via cross_val_score where supported; fallback to accuracy if AUC fails
    X_all = train_df[features].values
    y_all = train_df[cfg.target_col].values
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_state)
    fold_scores: List[float] = []
    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        clf_cv = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight=cfg.class_weight,
        )
        try:
            clf_cv.fit(X_tr, y_tr)
            probs_va = clf_cv.predict_proba(X_va)
            classes_cv = np.sort(np.unique(y_va))
            aucs = []
            # Map columns
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

    # Feature importances
    fi_df = None
    try:
        fi = getattr(clf, 'feature_importances_', None)
        if fi is not None:
            fi_df = pd.DataFrame({"feature": features, "importance": fi}).sort_values("importance", ascending=False)
    except Exception:
        fi_df = None

    return RFArtifacts(
        corr=corr,
        vif=vif,
        confusion=cm,
        roc_auc_macro=roc_macro,
        fold_scores=fold_scores,
        fpr=fpr_list,
        tpr=tpr_list,
        feature_importances_=fi_df,
    )


def save_artifacts(art: RFArtifacts, charts_dir: str = 'charts', metrics_dir: str = 'metrics') -> Dict[str, str]:
    import os
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    # Correlation heatmap
    if isinstance(art.corr, pd.DataFrame) and not art.corr.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(art.corr, annot=False, cmap='coolwarm', center=0)
        p = f"{charts_dir}/rf_correlations.png"
        plt.title('Feature Correlations (RF)')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['corr_heatmap'] = p

    # VIF barplot
    if isinstance(art.vif, pd.DataFrame) and not art.vif.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=art.vif, x='vif', y='feature', orient='h')
        p = f"{charts_dir}/rf_vif.png"
        plt.title('Variance Inflation Factors (RF)')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['vif'] = p

    # Feature importances
    if art.feature_importances_ is not None and not art.feature_importances_.empty:
        plt.figure(figsize=(8, 6))
        top = art.feature_importances_.head(20)
        sns.barplot(data=top, x='importance', y='feature', orient='h')
        p = f"{charts_dir}/rf_feature_importance.png"
        plt.title('Random Forest Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['feature_importance'] = p

    # Confusion matrix heatmap
    if isinstance(art.confusion, np.ndarray) and art.confusion.size:
        plt.figure(figsize=(6, 5))
        sns.heatmap(art.confusion, annot=True, fmt='d', cmap='Blues')
        p = f"{charts_dir}/rf_confusion.png"
        plt.title('Confusion Matrix (RF, Test)')
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
        title = 'RF ROC Curves'
        if art.roc_auc_macro is not None:
            title += f' (macro AUC={art.roc_auc_macro:.3f})'
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        p = f"{charts_dir}/rf_roc.png"
        plt.savefig(p)
        plt.close()
        paths['roc'] = p

    # Save textual metrics
    metrics_path = f"{metrics_dir}/rf_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write('Random Forest Metrics and Diagnostics\n')
        f.write('====================================\n')
        f.write(f"Macro ROC AUC (test): {art.roc_auc_macro}\n")
        if len(art.fold_scores):
            f.write(f"CV Macro AUC (mean): {np.nanmean(art.fold_scores):.4f}\n")
            f.write(f"CV Macro AUC (per fold): {art.fold_scores}\n")
        f.write('\nConfusion Matrix (rows=True, cols=Pred):\n')
        f.write(np.array2string(art.confusion))
        if art.feature_importances_ is not None and not art.feature_importances_.empty:
            f.write('\n\n--- Feature Importances (Top 50) ---\n')
            f.write(art.feature_importances_.head(50).to_string(index=False))
        if isinstance(art.vif, pd.DataFrame) and not art.vif.empty:
            f.write('\n\n--- VIF ---\n')
            f.write(art.vif.to_string(index=False))
        if isinstance(art.corr, pd.DataFrame) and not art.corr.empty:
            f.write('\n\n--- Correlations (head) ---\n')
            f.write(art.corr.round(3).to_string())
    paths['metrics_txt'] = metrics_path

    # Export CSVs
    if isinstance(art.vif, pd.DataFrame) and not art.vif.empty:
        vif_csv = f"{metrics_dir}/rf_vif.csv"
        art.vif.to_csv(vif_csv, index=False)
        paths['vif_csv'] = vif_csv
    if art.feature_importances_ is not None and not art.feature_importances_.empty:
        fi_csv = f"{metrics_dir}/rf_feature_importance.csv"
        art.feature_importances_.to_csv(fi_csv, index=False)
        paths['feature_importance_csv'] = fi_csv
    if isinstance(art.confusion, np.ndarray) and art.confusion.size:
        cm_csv = f"{metrics_dir}/rf_confusion_matrix.csv"
        pd.DataFrame(art.confusion).to_csv(cm_csv, index=False)
        paths['confusion_csv'] = cm_csv

    return paths


def run_rf_pipeline(df: pd.DataFrame, target_col: str = 'target') -> Tuple[RFArtifacts, Dict[str, str]]:
    cfg = RFConfig(target_col=target_col)
    features = [c for c in df.columns if c != target_col]
    train_df, test_df = split_data(df, cfg)
    artifacts = evaluate_model(train_df, test_df, features, cfg)
    paths = save_artifacts(artifacts, charts_dir='charts', metrics_dir='metrics')
    return artifacts, paths