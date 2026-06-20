"""
Optuna HPO for each model.
- 50 trials per model (configurable)
- 1-hour timeout per model
- Uses TimeSeriesSplit (3 folds) for cross-validation during HPO
- Results saved to hpo_results/ directory
- GPU-accelerated where supported

V2: Uses log1p target for regression, class weights for classification.
"""
import optuna
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
)
from src.config import *
from src.gpu_utils import create_xgb_params, create_lgbm_params, create_catboost_params

optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def _cv_score(model, X, y, task='regression', n_splits=HPO_CV_FOLDS):
    """
    Time-series cross-validation score.
    Returns negative RMSE (for regression) or F1 (for classification).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        if hasattr(X, 'iloc'):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
        
        if task == 'classification':
            if len(np.unique(y_tr)) < 2:
                # If training split only has 1 class, F1 is 0.0 (cannot fit binary classifier)
                scores.append(0.0)
                continue
                
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        if task == 'regression':
            scores.append(-np.sqrt(mean_squared_error(y_val, y_pred)))
        else:
            scores.append(f1_score(y_val, y_pred, zero_division=0))
    return np.mean(scores) if scores else (-999.0 if task == 'regression' else 0.0)


# ─────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE FUNCTIONS PER MODEL
# ─────────────────────────────────────────────────────────

def xgb_objective(trial, X, y, task='regression', class_weight_ratio=1.0):
    params = create_xgb_params({
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    })
    if task == 'regression':
        model = XGBRegressor(**params)
    else:
        params['eval_metric'] = 'logloss'
        params['scale_pos_weight'] = class_weight_ratio
        model = XGBClassifier(**params)
    return _cv_score(model, X, y, task)


def lgbm_objective(trial, X, y, task='regression', class_weight_ratio=1.0):
    params = create_lgbm_params({
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
    })
    if task == 'regression':
        model = LGBMRegressor(**params)
    else:
        params['scale_pos_weight'] = class_weight_ratio
        model = LGBMClassifier(**params)
    return _cv_score(model, X, y, task)


def catboost_objective(trial, X, y, task='regression', class_weight_ratio=1.0):
    params = create_catboost_params({
        'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'verbose': 0,
    })
    if task == 'regression':
        model = CatBoostRegressor(**params)
    else:
        params['auto_class_weights'] = 'Balanced'
        model = CatBoostClassifier(**params)
    return _cv_score(model, X, y, task)


def rf_objective(trial, X, y, task='regression', class_weight_ratio=1.0):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        'random_state': RANDOM_STATE, 'n_jobs': -1,
    }
    if task == 'regression':
        model = RandomForestRegressor(**params)
    else:
        params['class_weight'] = 'balanced'
        model = RandomForestClassifier(**params)
    return _cv_score(model, X, y, task)


def et_objective(trial, X, y, task='regression', class_weight_ratio=1.0):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
        'random_state': RANDOM_STATE, 'n_jobs': -1,
    }
    if task == 'regression':
        model = ExtraTreesRegressor(**params)
    else:
        params['class_weight'] = 'balanced'
        model = ExtraTreesClassifier(**params)
    return _cv_score(model, X, y, task)


# ─────────────────────────────────────────────────────────
# MASTER HPO RUNNER
# ─────────────────────────────────────────────────────────

HPO_OBJECTIVES = {
    'xgboost': xgb_objective,
    'lightgbm': lgbm_objective,
    'catboost': catboost_objective,
    'random_forest': rf_objective,
    'extra_trees': et_objective,
}


def run_hpo_for_model(model_name, X, y, task='regression', class_weight_ratio=1.0):
    """
    Run Optuna HPO for a single model.
    V2: Passes class_weight_ratio to objectives for classification.
    Returns best_params dict.
    Saves study to hpo_results/{model_name}_{task}.json
    """
    results_file = os.path.join(HPO_DIR, f'{model_name}_{task}_best.json')
    
    # Skip if already done
    if os.path.exists(results_file):
        logger.info(f"HPO already done for {model_name}_{task}. Loading from cache.")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    logger.info(f"Starting HPO for {model_name} ({task}) — {HPO_N_TRIALS} trials, "
                f"timeout={HPO_TIMEOUT_SECONDS}s")
    
    objective_fn = HPO_OBJECTIVES[model_name]
    
    # Higher is better for both: -RMSE (regression) or F1 (classification)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_fn(trial, X, y, task, class_weight_ratio),
        n_trials=HPO_N_TRIALS,
        timeout=HPO_TIMEOUT_SECONDS,
        show_progress_bar=False,
    )
    
    best_params = study.best_params
    best_value = study.best_value
    
    result = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials_completed': len(study.trials),
    }
    
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"HPO complete for {model_name}_{task}: "
                f"best_value={best_value:.4f}, trials={len(study.trials)}")
    
    return result


def run_all_hpo(X_train, y_train_reg, y_train_cls):
    """
    Run HPO for ALL models, both regression and classification.
    V2: Uses log1p target for regression HPO and class weights for classification HPO.
    Skips any model that's already been optimized (cached).
    """
    # Fix 6: Log-transform regression target for HPO to match training
    y_train_reg_log = np.log1p(y_train_reg)
    
    # Fix 3: Compute class weight ratio for HPO classifiers
    n_pos = y_train_cls.sum()
    n_neg = len(y_train_cls) - n_pos
    class_weight_ratio = n_neg / max(n_pos, 1)
    logger.info(f"HPO class weight ratio: {class_weight_ratio:.2f} (pos={n_pos}, neg={n_neg})")
    
    all_best = {}
    for name in HPO_OBJECTIVES:
        logger.info(f"\n{'='*60}")
        logger.info(f"HPO: {name}")
        logger.info(f"{'='*60}")
        
        try:
            reg_result = run_hpo_for_model(name, X_train, y_train_reg_log, 'regression')
            cls_result = run_hpo_for_model(name, X_train, y_train_cls, 'classification',
                                           class_weight_ratio=class_weight_ratio)
            all_best[name] = {'reg': reg_result, 'cls': cls_result}
        except Exception as e:
            logger.error(f"HPO FAILED for {name}: {e}")
            logger.info("Using default params instead.")
            all_best[name] = None
    
    return all_best
