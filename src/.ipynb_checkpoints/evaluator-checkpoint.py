"""
Evaluation metrics, leaderboards, and comparisons.

V2 IMPROVEMENTS (Fix 7):
- Precision@K and Recall@K for operationally relevant hotspot evaluation
- MAPE on non-zero observations for meaningful regression quality
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from src.config import logger

def evaluate_regressor(y_true, y_pred):
    """
    Computes regression metrics.
    V2: Also computes MAPE on non-zero observations.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Fix 7: MAPE on non-zero observations only (where prediction quality matters)
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape_nonzero = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape_nonzero = float('nan')
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R²': float(r2),
        'MAPE_nonzero': float(mape_nonzero),
    }


def evaluate_classifier(y_true, y_pred, y_prob):
    """
    Computes classification metrics.
    V2: Also computes Precision@K and Recall@K.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5  # Fallback if only 1 class is present in y_true
    
    # Fix 7: Precision@K — "of the top K predicted hotspots, how many are actual?"
    # This is the operationally relevant metric for patrol dispatch.
    p_at_100 = _precision_at_k(y_true, y_prob, k=100)
    p_at_500 = _precision_at_k(y_true, y_prob, k=500)
    r_at_100 = _recall_at_k(y_true, y_prob, k=100)
        
    return {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'ROC_AUC': float(auc),
        'P@100': float(p_at_100),
        'P@500': float(p_at_500),
        'R@100': float(r_at_100),
    }


def _precision_at_k(y_true, y_prob, k=100):
    """Precision@K: of the top-k by predicted probability, how many are actually positive?"""
    if len(y_true) == 0 or k <= 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_indices = np.argsort(y_prob)[-k:]
    return float(y_true[top_k_indices].sum()) / k


def _recall_at_k(y_true, y_prob, k=100):
    """Recall@K: of all actual positives, how many are in the top-k predictions?"""
    n_pos = y_true.sum()
    if n_pos == 0 or k <= 0:
        return 0.0
    k = min(k, len(y_true))
    top_k_indices = np.argsort(y_prob)[-k:]
    return float(y_true[top_k_indices].sum()) / n_pos


def build_leaderboard(results, y_true_reg, y_true_cls):
    """
    Builds dataframes comparing all models.
    V2: Includes P@100, P@500, R@100 and MAPE_nonzero.
    """
    reg_rows = []
    cls_rows = []
    
    for name, preds in results.items():
        if 'y_pred_reg' in preds:
            reg_m = evaluate_regressor(y_true_reg, preds['y_pred_reg'])
            reg_m['Model'] = name
            reg_rows.append(reg_m)
            
        if 'y_pred_cls' in preds:
            cls_m = evaluate_classifier(y_true_cls, preds['y_pred_cls'], preds['y_prob_cls'])
            cls_m['Model'] = name
            cls_rows.append(cls_m)
            
    df_reg = pd.DataFrame(reg_rows)
    if not df_reg.empty:
        df_reg = df_reg[['Model', 'R²', 'RMSE', 'MAE', 'MSE', 'MAPE_nonzero']].sort_values(by='R²', ascending=False)
        
    df_cls = pd.DataFrame(cls_rows)
    if not df_cls.empty:
        df_cls = df_cls[['Model', 'F1', 'ROC_AUC', 'P@100', 'P@500', 'R@100', 
                         'Accuracy', 'Precision', 'Recall']].sort_values(by='F1', ascending=False)
        
    return df_reg, df_cls


def build_base_vs_enhanced_comparison(base_results, enhanced_results, y_true_reg, y_true_cls):
    """
    Builds a summary comparison of Base vs Enhanced models.
    """
    base_reg, base_cls = build_leaderboard(base_results, y_true_reg, y_true_cls)
    enh_reg, enh_cls = build_leaderboard(enhanced_results, y_true_reg, y_true_cls)
    
    logger.info("=== BASE VS ENHANCED REGRESSION COMPARISON ===")
    merged_reg = pd.merge(base_reg[['Model', 'R²', 'RMSE']], enh_reg[['Model', 'R²', 'RMSE']], 
                          on='Model', suffixes=('_base', '_enhanced'))
    logger.info(f"\n{merged_reg.to_string()}")
    
    logger.info("=== BASE VS ENHANCED CLASSIFICATION COMPARISON ===")
    merged_cls = pd.merge(base_cls[['Model', 'F1', 'ROC_AUC', 'P@100']], 
                          enh_cls[['Model', 'F1', 'ROC_AUC', 'P@100']], 
                          on='Model', suffixes=('_base', '_enhanced'))
    logger.info(f"\n{merged_cls.to_string()}")
    
    return merged_reg, merged_cls
