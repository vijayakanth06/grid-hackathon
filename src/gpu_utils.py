"""
GPU-aware model factory.
Creates model instances with GPU params if available, CPU otherwise.
Automatically handles the XGBoost/LightGBM/CatBoost GPU differences.
"""
from src.config import *

def create_xgb_params(trial_params: dict = None) -> dict:
    """
    Create XGBoost params with GPU if available.
    If trial_params provided (from Optuna), override defaults.
    """
    base = {
        'tree_method': XGB_TREE_METHOD,
        'device': XGB_DEVICE,
        'random_state': RANDOM_STATE,
        'n_jobs': -1 if not HAS_GPU else 1,  # GPU uses 1 job
        'verbosity': 0,
    }
    if trial_params:
        base.update(trial_params)
    else:
        base.update({
            'n_estimators': 500, 'max_depth': 8,
            'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_weight': 5,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
        })
    return base


def create_lgbm_params(trial_params: dict = None) -> dict:
    """LightGBM with GPU if available."""
    base = {
        'device': LGBM_DEVICE,
        'random_state': RANDOM_STATE,
        'n_jobs': -1, 'verbose': -1,
    }
    if LGBM_DEVICE == 'gpu':
        base['gpu_use_dp'] = True  # V100 supports double precision
    if trial_params:
        base.update(trial_params)
    else:
        base.update({
            'n_estimators': 500, 'max_depth': 8,
            'learning_rate': 0.05, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'min_child_samples': 20,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
        })
    return base


def create_catboost_params(trial_params: dict = None) -> dict:
    """CatBoost with GPU if available."""
    base = {
        'task_type': CATBOOST_TASK,
        'random_seed': RANDOM_STATE,
        'verbose': 100,
    }
    if CATBOOST_TASK == 'GPU':
        base['devices'] = '0'  # Use GPU 0
    if trial_params:
        base.update(trial_params)
    else:
        base.update({
            'iterations': 500, 'depth': 8,
            'learning_rate': 0.05, 'l2_leaf_reg': 3,
        })
    return base
