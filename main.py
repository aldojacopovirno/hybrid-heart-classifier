from __future__ import annotations

import pandas as pd

from script.encoder import encode
from script.processing import process
from script.onehot_encoding import onehot_encode
from script.eda import run_full_eda
from script.olr_model import run_olr_pipeline
from script.rf_model import run_rf_pipeline
from script.xgb_model import run_xgb_pipeline
from script.mlp_model import run_mlp_pipeline
from typing import Dict, Any, List
import numpy as np

def load_and_encode(csv_path: str = "data/heart_disease_uci.csv") -> pd.DataFrame:
    """Load the raw dataset and convert categorical fields to numeric codes.

    Parameters
    ----------
    csv_path : str, optional
        Path to the UCI heart disease CSV file.

    Returns
    -------
    pandas.DataFrame
        Encoded dataset ready for downstream processing.
    """

    df = pd.read_csv(csv_path)
    encoded_data = encode(df)
    return encoded_data

if __name__ == "__main__":
    encoded_data = load_and_encode()
    # Process dataset (imputations + scaling)
    processed_data = process(encoded_data)
    # Apply one-hot encoding for 'thal' into boolean features (model input)
    onehot_data = onehot_encode(processed_data)

    print("Encoded data preview:")
    print(encoded_data.head())

    print("\nProcessed data preview:")
    print(processed_data.head())

    print("\nOne-hot data preview:")
    print(onehot_data.head())

    print("\nRunning EDA...")
    # Use processed dataset for EDA
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

    # Run OLR pipeline on one-hot dataset and save performance
    print("\nTraining OLR model and evaluating...")
    olr_art, saved = run_olr_pipeline(onehot_data, target_col="target")
    print("Saved OLR artifacts (charts + metrics):")
    for k, v in saved.items():
        print(f"- {k}: {v}")

    # Run Random Forest pipeline on one-hot dataset and save performance
    print("\nTraining Random Forest model and evaluating...")
    rf_art, rf_saved = run_rf_pipeline(onehot_data, target_col="target")
    print("Saved RF artifacts (charts + metrics):")
    for k, v in rf_saved.items():
        print(f"- {k}: {v}")

    # Run XGBoost pipeline on one-hot dataset and save performance
    print("\nTraining XGBoost model and evaluating...")
    xgb_art, xgb_saved = run_xgb_pipeline(onehot_data, target_col="target")
    print("Saved XGB artifacts (charts + metrics):")
    for k, v in xgb_saved.items():
        print(f"- {k}: {v}")

    # Run MLP pipeline on one-hot dataset and save performance
    print("\nTraining MLP model and evaluating...")
    mlp_art, mlp_saved = run_mlp_pipeline(onehot_data, target_col="target")
    print("Saved MLP artifacts (charts + metrics):")
    for k, v in mlp_saved.items():
        print(f"- {k}: {v}")

    # Build comparative metrics table across models
    print("\nComparing models with common metrics...")
    def summarize_model(name: str, art: Any) -> Dict[str, Any]:
        """Aggregate common metrics from a model artifact.

        Parameters
        ----------
        name : str
            Identifier for the model being summarized.
        art : Any
            Artifact namespace with evaluation attributes such as confusion
            matrix and ROC AUC scores.

        Returns
        -------
        dict
            Dictionary collecting macro and weighted metrics plus confusion
            matrix statistics.
        """

        # Common metrics + additional: AUC (macro), CV AUC mean, Accuracy, Precision/Recall/F1 (macro & weighted),
        # per-class support, MCC, balanced_accuracy, specificity_macro.
        cm = getattr(art, 'confusion', None)
        test_auc = getattr(art, 'roc_auc_macro', None)
        cv_scores = getattr(art, 'fold_scores', []) or []
        cv_mean = float(np.nanmean(cv_scores)) if len(cv_scores) else np.nan
        accuracy = np.nan
        precision_macro = np.nan
        recall_macro = np.nan
        f1_macro = np.nan
        precision_weighted = np.nan
        recall_weighted = np.nan
        f1_weighted = np.nan
        mcc = np.nan
        balanced_accuracy = np.nan
        specificity_macro = np.nan
        n_classes = np.nan
        support_total = np.nan
        if isinstance(cm, np.ndarray) and cm.size:
            total = cm.sum()
            correct = np.trace(cm)
            accuracy = float(correct / total) if total > 0 else np.nan
            # Per-class precision/recall
            with np.errstate(divide='ignore', invalid='ignore'):
                tp = np.diag(cm).astype(float)
                fp = cm.sum(axis=0).astype(float) - tp
                fn = cm.sum(axis=1).astype(float) - tp
                tn = total - (tp + fp + fn)
                prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
                rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
                spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
                precision_macro = float(np.nanmean(prec))
                recall_macro = float(np.nanmean(rec))
                f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), np.nan)
                f1_macro = float(np.nanmean(f1))
                # Weighted by support (true instances per class)
                support = cm.sum(axis=1).astype(float)
                support_total = float(np.nansum(support))
                if support_total > 0:
                    weights = support / support_total
                    precision_weighted = float(np.nansum(prec * weights))
                    recall_weighted = float(np.nansum(rec * weights))
                    f1_weighted = float(np.nansum(f1 * weights))
                    specificity_macro = float(np.nanmean(spec))
                # Matthews Correlation Coefficient (multiclass): Gorodkin generalization
                # mcc = (c*sum(tp) - sum_i P_i T_i) / sqrt((c*sum P_i - sum P_i^2) * (c*sum T_i - sum T_i^2))
                c = cm.shape[0]
                n = total
                if c > 1 and n > 0:
                    sum_tp = float(tp.sum())
                    P = cm.sum(axis=0).astype(float)
                    T = cm.sum(axis=1).astype(float)
                    sum_p_t = float(np.dot(P, T))
                    denom_left = c * float((P).sum()) - float((P**2).sum())
                    denom_right = c * float((T).sum()) - float((T**2).sum())
                    denom = np.sqrt(denom_left * denom_right)
                    num = c * sum_tp - sum_p_t
                    mcc = float(num / denom) if denom > 0 else np.nan
                    # Balanced accuracy
                    balanced_accuracy = float(np.nanmean(rec))
                n_classes = float(c)
        return {
            'model': name,
            'test_macro_auc': test_auc,
            'cv_macro_auc_mean': cv_mean,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'specificity_macro': specificity_macro,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'n_classes': n_classes,
            'support_total': support_total,
        }

    rows: List[Dict[str, Any]] = [
        summarize_model('OLR', olr_art),
        summarize_model('RandomForest', rf_art),
        summarize_model('XGBoost', xgb_art),
        summarize_model('MLP', mlp_art),
    ]
    comp_df = pd.DataFrame(rows)
    # Order columns
    comp_df = comp_df[
        [
            'model',
            'test_macro_auc', 'cv_macro_auc_mean', 'accuracy',
            'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'specificity_macro', 'balanced_accuracy', 'mcc',
            'n_classes', 'support_total'
        ]
    ]
    print("\nComparative performance table:")
    print(comp_df.round(4).to_string(index=False))
    # Save to metrics directory
    try:
        comp_df.to_csv('metrics/comparative_metrics.csv', index=False)
        print("\nSaved comparative table to metrics/comparative_metrics.csv")
    except Exception as e:
        print(f"[WARN] Could not save comparative table: {e}")
