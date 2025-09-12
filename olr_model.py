from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor


# -------------------------
# Data diagnostics helpers
# -------------------------

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    return corr


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    X = df[features].copy()
    X = sm.add_constant(X, has_constant='add')
    vif_data = []
    for i, col in enumerate(X.columns):
        if col == 'const':
            continue
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({'feature': col, 'vif': float(vif)})
    return pd.DataFrame(vif_data).sort_values('vif', ascending=False).reset_index(drop=True)


def box_tidwell_test(df: pd.DataFrame, y_col: str, cont_features: List[str]) -> pd.DataFrame:
    """
    Box-Tidwell for ordinal logit (approx) by testing added terms x*log(x).
    Note: Only for strictly positive continuous predictors.
    """
    results = []
    # Ensure positive by small shift if needed
    eps = 1e-6
    y = df[y_col]
    for col in cont_features:
        x = df[col].astype(float)
        minx = x.min()
        if minx <= 0:
            x = x - minx + eps
        z = x * np.log(x)
        X = pd.DataFrame({col: x, f"{col}_log": z})
        mod = OrderedModel(y, X, distr='logit')
        try:
            res = mod.fit(method='bfgs', disp=False)
            pval = res.pvalues.get(f"{col}_log", np.nan)
        except Exception:
            pval = np.nan
        results.append({'feature': col, 'p_value': pval})
    return pd.DataFrame(results)


def brant_test(model_result) -> pd.DataFrame:
    """
    Placeholder Brant test approximation: we compare coefficients across
    thresholds by fitting separate binary logits (cumulative splits) and
    testing equality via simple variance-based z. This is a heuristic
    since statsmodels doesn't expose a direct Brant test.
    """
    endog = model_result.model.endog
    exog = model_result.model.exog
    exog_names = model_result.model.exog_names

    # For cumulative logit, create K-1 binary splits
    y = pd.Series(endog)
    levels = sorted(pd.unique(y))
    coef_list = []
    for thr in levels[:-1]:
        yy = (y > thr).astype(int)
        logit = sm.Logit(yy, exog)
        try:
            fit = logit.fit(disp=False)
            coef_list.append(pd.Series(fit.params, index=exog_names))
        except Exception:
            coef_list.append(pd.Series(np.nan, index=exog_names))

    if not coef_list:
        return pd.DataFrame()
    coefs = pd.concat(coef_list, axis=1)
    coefs.columns = [f"threshold_{i+1}" for i in range(coefs.shape[1])]
    # Compute simple variance across thresholds as a proxy statistic
    stat = coefs.var(axis=1)
    return pd.DataFrame({'feature': stat.index, 'coef_variance': stat.values})


# -------------------------
# Modeling helpers
# -------------------------

@dataclass
class OLRConfig:
    target_col: str = 'target'
    test_size: float = 0.2
    random_state: int = 42
    k_folds: int = 5
    distr: str = 'logit'  # 'logit' or 'probit'


@dataclass
class OLRArtifacts:
    corr: pd.DataFrame
    vif: pd.DataFrame
    box_tidwell: pd.DataFrame
    brant: pd.DataFrame
    confusion: np.ndarray
    roc_auc_macro: Optional[float]
    fold_scores: List[float]
    thresholds: np.ndarray
    fpr: List[np.ndarray]
    tpr: List[np.ndarray]


def split_data(df: pd.DataFrame, cfg: OLRConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_state, stratify=df[cfg.target_col]
    )
    return train_df, test_df


def log_transform_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Apply log(1 + x_shifted) to all non-target columns.
    For any column with min <= 0, shift by (1 - min) to ensure positivity.
    Works only on numeric columns; non-numerics (if any) are ignored.
    """
    df_t = df.copy()
    for col in df_t.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df_t[col]):
            col_min = df_t[col].min()
            shift = 1 - col_min if pd.notna(col_min) and col_min <= 0 else 0
            df_t[col] = np.log1p(df_t[col] + shift)
    return df_t


def fit_olr(train_df: pd.DataFrame, features: List[str], cfg: OLRConfig):
    y = train_df[cfg.target_col]
    X = train_df[features]
    model = OrderedModel(y, X, distr=cfg.distr)
    res = model.fit(method='bfgs', disp=False)
    return model, res


def predict_prob(model: OrderedModel, res, X: pd.DataFrame) -> np.ndarray:
    pred = res.model.predict(res.params, exog=X)
    # returns probabilities for each class
    return np.asarray(pred)


def evaluate_model(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], cfg: OLRConfig) -> OLRArtifacts:
    # Global log transformation on all non-target features before anything else
    train_df = log_transform_features(train_df, cfg.target_col)
    test_df = log_transform_features(test_df, cfg.target_col)
    # Recompute features list (structure unchanged)
    features = [c for c in train_df.columns if c != cfg.target_col]

    # Diagnostics
    corr = compute_correlations(train_df[features + [cfg.target_col]])
    vif = compute_vif(train_df, features)
    # Continuous predictors heuristic: numeric and more than 10 unique values
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(train_df[c])]
    cont_cols = [c for c in numeric_cols if train_df[c].nunique() > 10]
    box_tid = box_tidwell_test(train_df[[cfg.target_col] + cont_cols + []], cfg.target_col, cont_cols) if cont_cols else pd.DataFrame()

    # Fit final model on train
    model, res = fit_olr(train_df, features, cfg)

    # Brant approximation
    brant = brant_test(res)

    # Confusion matrix using argmax of predicted probabilities
    X_test = test_df[features]
    y_test = test_df[cfg.target_col].values
    probs = predict_prob(model, res, X_test)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # ROC: One-vs-rest per class, macro average
    classes = np.sort(np.unique(y_test))
    fpr_list, tpr_list, roc_auc_list = [], [], []
    for k in classes:
        y_true_bin = (y_test == k).astype(int)
        y_score = probs[:, int(k)] if int(k) < probs.shape[1] else probs[:, -1]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(auc(fpr, tpr))
    roc_macro = float(np.mean(roc_auc_list)) if roc_auc_list else None

    # K-fold CV on training set (macro AUC)
    kf = KFold(n_splits=cfg.k_folds, shuffle=True, random_state=cfg.random_state)
    fold_scores: List[float] = []
    X_all = train_df[features].reset_index(drop=True)
    y_all = train_df[cfg.target_col].reset_index(drop=True)
    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
        y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
        mod = OrderedModel(y_tr, X_tr, distr=cfg.distr)
        try:
            res_cv = mod.fit(method='bfgs', disp=False)
            probs_va = np.asarray(res_cv.model.predict(res_cv.params, exog=X_va))
            classes_cv = np.sort(np.unique(y_va))
            aucs = []
            for k in classes_cv:
                y_true_bin = (y_va.values == k).astype(int)
                y_score = probs_va[:, int(k)] if int(k) < probs_va.shape[1] else probs_va[:, -1]
                fpr, tpr, _ = roc_curve(y_true_bin, y_score)
                aucs.append(auc(fpr, tpr))
            fold_scores.append(float(np.mean(aucs)))
        except Exception:
            fold_scores.append(np.nan)

    # attach fitted result for detailed logs downstream
    evaluate_model.last_fit_result = res  # type: ignore[attr-defined]
    evaluate_model.features = features  # type: ignore[attr-defined]

    return OLRArtifacts(
        corr=corr,
        vif=vif,
        box_tidwell=box_tid,
        brant=brant,
        confusion=cm,
        roc_auc_macro=roc_macro,
        fold_scores=fold_scores,
        thresholds=np.array([]),
        fpr=fpr_list,
        tpr=tpr_list,
    )


def save_artifacts(art: OLRArtifacts, charts_dir: str = 'charts', metrics_dir: str = 'metrics') -> Dict[str, str]:
    import os
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    paths: Dict[str, str] = {}

    # Correlation heatmap
    if not art.corr.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(art.corr, annot=False, cmap='coolwarm', center=0)
        p = f"{charts_dir}/olr_correlations.png"
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['corr_heatmap'] = p

    # VIF barplot
    if not art.vif.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=art.vif, x='vif', y='feature', orient='h')
        p = f"{charts_dir}/olr_vif.png"
        plt.title('Variance Inflation Factors')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['vif'] = p

    # Brant proxy barplot
    if not art.brant.empty:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=art.brant, x='coef_variance', y='feature', orient='h')
        p = f"{charts_dir}/olr_brant_proxy.png"
        plt.title('Brant Test Proxy (Coef Variance)')
        plt.tight_layout()
        plt.savefig(p)
        plt.close()
        paths['brant_proxy'] = p

    # Confusion matrix heatmap
    if art.confusion.size:
        plt.figure(figsize=(6, 5))
        sns.heatmap(art.confusion, annot=True, fmt='d', cmap='Blues')
        p = f"{charts_dir}/olr_confusion.png"
        plt.title('Confusion Matrix (Test)')
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
        plt.title(f'ROC Curves (macro AUC={art.roc_auc_macro:.3f})' if art.roc_auc_macro is not None else 'ROC Curves')
        plt.legend()
        plt.tight_layout()
        p = f"{charts_dir}/olr_roc.png"
        plt.savefig(p)
        plt.close()
        paths['roc'] = p

    # Save textual metrics (detailed)
    metrics_path = f"{metrics_dir}/olr_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write('OLR Metrics and Diagnostics\n')
        f.write('===========================\n')
        f.write(f"Macro ROC AUC (test): {art.roc_auc_macro}\n")
        if len(art.fold_scores):
            f.write(f"CV Macro AUC (mean): {np.nanmean(art.fold_scores):.4f}\n")
            f.write(f"CV Macro AUC (per fold): {art.fold_scores}\n")
        f.write('\nConfusion Matrix (rows=True, cols=Pred):\n')
        f.write(np.array2string(art.confusion))
        f.write('\n\n--- Detailed Coefficients (OLR) ---\n')
        # Try to access last fitted result populated by evaluate_model
        res = getattr(evaluate_model, 'last_fit_result', None)
        feats = getattr(evaluate_model, 'features', None)
        if res is not None:
            try:
                # Full statsmodels summary
                f.write('\n\n--- statsmodels Summary ---\n')
                f.write(str(res.summary()))
            except Exception:
                f.write('\n[WARN] Unable to serialize full result stats.')
        # Box-Tidwell details
        if not art.box_tidwell.empty:
            f.write('\n\n--- Box-Tidwell (x*log(x)) p-values ---\n')
            f.write(art.box_tidwell.to_string(index=False))
        # Brant proxy details
        if not art.brant.empty:
            f.write('\n\n--- Brant Test Proxy (coef variance across thresholds) ---\n')
            f.write(art.brant.to_string(index=False))
        # VIF details
        if not art.vif.empty:
            f.write('\n\n--- VIF ---\n')
            f.write(art.vif.to_string(index=False))
        # Correlation matrix snapshot
        if not art.corr.empty:
            f.write('\n\n--- Correlations (head) ---\n')
            f.write(art.corr.round(3).to_string())
    paths['metrics_txt'] = metrics_path

    # Also export CSVs for tables to metrics_dir
    if not art.vif.empty:
        vif_csv = f"{metrics_dir}/vif.csv"
        art.vif.to_csv(vif_csv, index=False)
        paths['vif_csv'] = vif_csv
    if not art.box_tidwell.empty:
        bt_csv = f"{metrics_dir}/box_tidwell.csv"
        art.box_tidwell.to_csv(bt_csv, index=False)
        paths['box_tidwell_csv'] = bt_csv
    if not art.brant.empty:
        brant_csv = f"{metrics_dir}/brant_proxy.csv"
        art.brant.to_csv(brant_csv, index=False)
        paths['brant_proxy_csv'] = brant_csv
    # Confusion matrix
    if art.confusion.size:
        cm_csv = f"{metrics_dir}/confusion_matrix.csv"
        pd.DataFrame(art.confusion).to_csv(cm_csv, index=False)
        paths['confusion_csv'] = cm_csv

    return paths


def run_olr_pipeline(df: pd.DataFrame, target_col: str = 'target') -> Tuple[OLRArtifacts, Dict[str, str]]:
    cfg = OLRConfig(target_col=target_col)
    # Define features: all except target
    features = [c for c in df.columns if c != target_col]
    train_df, test_df = split_data(df, cfg)
    artifacts = evaluate_model(train_df, test_df, features, cfg)
    paths = save_artifacts(artifacts, charts_dir='charts', metrics_dir='metrics')
    return artifacts, paths
