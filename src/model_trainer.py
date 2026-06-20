"""
Fault-tolerant multi-model training pipeline.

KEY FEATURES:
1. Checks checkpoint before training each model → skip if already done
2. GPU auto-detection → XGBoost gpu_hist, LightGBM gpu, CatBoost GPU
3. Uses HPO-optimized params if available, else defaults
4. try/except around EACH model → one failure doesn't kill the pipeline
5. Logs everything to logs/pipeline.log
6. Saves predictions alongside models for later comparison

V2 IMPROVEMENTS:
- Fix 3: Class weights for imbalanced classification
- Fix 5: TabM receives DataFrames with column names
- Fix 6: Log-transform regression target for better distribution
"""
import joblib, json, os, traceback
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    StackingRegressor, StackingClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from src.config import *
from src.gpu_utils import create_xgb_params, create_lgbm_params, create_catboost_params
from src.checkpoint import is_model_done, mark_model_done

# ─────────────────────────────────────────────────────────
# MODEL FACTORY — Creates model instances with HPO or default params
# ─────────────────────────────────────────────────────────

def _create_model(name, task, hpo_params=None, class_weight_ratio=1.0):
    """
    Create a model instance.
    If hpo_params provided → use optimized hyperparameters.
    If not → use sensible defaults.
    GPU auto-detected from config.
    
    Fix 3: class_weight_ratio = n_neg / n_pos for scale_pos_weight in classifiers.
    """
    params = hpo_params or {}
    
    if name == 'xgboost':
        full_params = create_xgb_params(params)
        if task == 'cls':
            full_params['scale_pos_weight'] = class_weight_ratio
            return XGBClassifier(**full_params, eval_metric='logloss')
        return XGBRegressor(**full_params)
    
    elif name == 'lightgbm':
        full_params = create_lgbm_params(params)
        if task == 'cls':
            full_params['scale_pos_weight'] = class_weight_ratio
            return LGBMClassifier(**full_params)
        return LGBMRegressor(**full_params)
    
    elif name == 'catboost':
        full_params = create_catboost_params(params)
        if task == 'cls':
            full_params['auto_class_weights'] = 'Balanced'
            return CatBoostClassifier(**full_params)
        return CatBoostRegressor(**full_params)
    
    elif name == 'random_forest':
        defaults = {'n_estimators': 300, 'max_depth': 15, 'min_samples_leaf': 10,
                     'random_state': RANDOM_STATE, 'n_jobs': -1}
        defaults.update(params)
        if task == 'cls':
            defaults['class_weight'] = 'balanced'
            return RandomForestClassifier(**defaults)
        return RandomForestRegressor(**defaults)
    
    elif name == 'extra_trees':
        defaults = {'n_estimators': 300, 'max_depth': 15, 'min_samples_leaf': 10,
                     'random_state': RANDOM_STATE, 'n_jobs': -1}
        defaults.update(params)
        if task == 'cls':
            defaults['class_weight'] = 'balanced'
            return ExtraTreesClassifier(**defaults)
        return ExtraTreesRegressor(**defaults)


# ─────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE — FAULT TOLERANT
# ─────────────────────────────────────────────────────────

MODEL_NAMES = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'extra_trees']


def temporal_split(df, split_date=TRAIN_TEST_SPLIT_DATE):
    df['date'] = pd.to_datetime(df['date'])
    split = pd.Timestamp(split_date)
    return df[df['date'] < split], df[df['date'] >= split]


def train_all_models(df, feature_cols, model_prefix='base', hpo_results=None):
    """
    Train all models with fault tolerance.
    
    V2: Uses log-transform for regression target and class weights for classification.
    
    FAULT TOLERANCE:
    - Each model wrapped in try/except
    - If model already trained (checkpoint exists) → SKIP
    - If model crashes → log error, continue to next model
    - Predictions saved to disk alongside models
    
    Args:
        df: DataFrame with features + targets
        feature_cols: Feature column names
        model_prefix: 'base' or 'enhanced'
        hpo_results: dict from run_all_hpo() or None for defaults
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    train, test = temporal_split(df)
    
    X_train = train[feature_cols].values
    X_test = test[feature_cols].values
    y_train_reg_raw = train['violation_count'].values
    y_test_reg_raw = test['violation_count'].values
    y_train_cls = train['is_hotspot'].values
    y_test_cls = test['is_hotspot'].values
    
    # ── Fix 6: LOG-TRANSFORM regression target ──────────────────
    # np.log1p(x) = log(1+x), handles zeros gracefully
    y_train_reg = np.log1p(y_train_reg_raw).astype(np.float32)
    y_test_reg = np.log1p(y_test_reg_raw).astype(np.float32)
    logger.info(f"Log-transform applied: violation_count range [{y_train_reg_raw.min()}, {y_train_reg_raw.max()}] "
                f"→ log1p range [{y_train_reg.min():.2f}, {y_train_reg.max():.2f}]")
    
    # ── Fix 3: Compute class weight ratio ───────────────────────
    n_pos = y_train_cls.sum()
    n_neg = len(y_train_cls) - n_pos
    class_weight_ratio = n_neg / max(n_pos, 1)
    logger.info(f"Class balance: {n_pos} hotspots / {n_neg} non-hotspots "
                f"(ratio={class_weight_ratio:.2f})")
    
    # Save test data for later comparison (save RAW values for evaluation)
    np.save(os.path.join(MODEL_DIR, f'{model_prefix}_y_test_reg.npy'), y_test_reg_raw)
    np.save(os.path.join(MODEL_DIR, f'{model_prefix}_y_test_cls.npy'), y_test_cls)
    
    # ── Fix 5: Keep DataFrames for TabM ─────────────────────────
    X_train_df = train[feature_cols].copy()
    X_test_df = test[feature_cols].copy()
    
    results = {}
    
    # ── Train each model independently ───────────────────
    for name in MODEL_NAMES:
        # ─── CHECK CHECKPOINT ────────────────────────────
        if is_model_done(name, model_prefix):
            logger.info(f"⏭️  SKIP {name} ({model_prefix}) — already completed")
            # Load saved predictions
            try:
                preds = np.load(os.path.join(MODEL_DIR, f'{model_prefix}_{name}_preds.npz'))
                results[name] = {
                    'y_pred_reg': preds['y_pred_reg'],
                    'y_pred_cls': preds['y_pred_cls'],
                    'y_prob_cls': preds['y_prob_cls'],
                }
            except FileNotFoundError:
                logger.warning(f"Predictions file missing for {name}, will retrain")
            else:
                continue
        
        # ─── TRAIN MODEL ────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 Training: {name} ({model_prefix})")
        logger.info(f"{'='*60}")
        
        try:
            # Get HPO params if available
            hpo_reg_params = None
            hpo_cls_params = None
            if hpo_results and name in hpo_results and hpo_results[name]:
                hpo_reg_params = hpo_results[name]['reg']['best_params']
                hpo_cls_params = hpo_results[name]['cls']['best_params']
                logger.info(f"Using HPO-optimized params for {name}")
            else:
                logger.info(f"Using default params for {name}")
            
            # — REGRESSOR (trained on log1p target) —
            reg = _create_model(name, 'reg', hpo_reg_params)
            if name == 'xgboost':
                reg.fit(X_train, y_train_reg,
                        eval_set=[(X_test, y_test_reg)], verbose=100)
            elif name == 'lightgbm':
                reg.fit(X_train, y_train_reg,
                        eval_set=[(X_test, y_test_reg)])
            elif name == 'catboost':
                reg.fit(X_train, y_train_reg, eval_set=(X_test, y_test_reg))
            else:
                reg.fit(X_train, y_train_reg)
            
            # Fix 6: Inverse log transform predictions back to original scale
            y_pred_reg_log = reg.predict(X_test)
            y_pred_reg = np.expm1(y_pred_reg_log)
            y_pred_reg = np.maximum(y_pred_reg, 0)  # Clamp negatives to 0
            
            # — CLASSIFIER (with class weights, Fix 3) —
            cls = _create_model(name, 'cls', hpo_cls_params, class_weight_ratio)
            if name == 'xgboost':
                cls.fit(X_train, y_train_cls,
                        eval_set=[(X_test, y_test_cls)], verbose=100)
            elif name == 'lightgbm':
                cls.fit(X_train, y_train_cls,
                        eval_set=[(X_test, y_test_cls)])
            elif name == 'catboost':
                cls.fit(X_train, y_train_cls, eval_set=(X_test, y_test_cls))
            else:
                cls.fit(X_train, y_train_cls)
            y_pred_cls = cls.predict(X_test)
            y_prob_cls = cls.predict_proba(X_test)[:, 1]
            
            # — SAVE —
            joblib.dump(reg, os.path.join(MODEL_DIR, f'{model_prefix}_{name}_reg.joblib'))
            joblib.dump(cls, os.path.join(MODEL_DIR, f'{model_prefix}_{name}_cls.joblib'))
            np.savez(
                os.path.join(MODEL_DIR, f'{model_prefix}_{name}_preds.npz'),
                y_pred_reg=y_pred_reg, y_pred_cls=y_pred_cls, y_prob_cls=y_prob_cls
            )
            
            # — CHECKPOINT — evaluate against RAW target values
            from src.evaluator import evaluate_regressor, evaluate_classifier
            metrics = {
                'reg': evaluate_regressor(y_test_reg_raw, y_pred_reg),
                'cls': evaluate_classifier(y_test_cls, y_pred_cls, y_prob_cls),
            }
            mark_model_done(name, model_prefix, metrics)
            
            results[name] = {
                'y_pred_reg': y_pred_reg,
                'y_pred_cls': y_pred_cls,
                'y_prob_cls': y_prob_cls,
            }
            
            logger.info(f"✅ {name} complete: R²={metrics['reg']['R²']:.4f}, "
                        f"F1={metrics['cls']['F1']:.4f}, ROC_AUC={metrics['cls']['ROC_AUC']:.4f}")
        
        except Exception as e:
            logger.error(f"❌ {name} FAILED: {e}")
            logger.error(traceback.format_exc())
            logger.info("Continuing to next model...")
    
    # ── TabM (optional) — Fix 5: Pass DataFrames ────────────────
    if not is_model_done('tabm', model_prefix):
        try:
            from deeptab.models import TabMClassifier, TabMRegressor
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 Training: TabM ({model_prefix})")
            logger.info(f"{'='*60}")
            # Fix 5: TabM/deeptab expects DataFrame with column names, not numpy arrays
            tabm_reg = TabMRegressor()
            tabm_reg.fit(X_train_df.astype(np.float32), pd.Series(y_train_reg, name='target'),
                         max_epochs=100, lr=0.01)
            y_pred_reg_log = tabm_reg.predict(X_test_df.astype(np.float32))
            y_pred_reg = np.maximum(np.expm1(y_pred_reg_log), 0)
            
            tabm_cls = TabMClassifier()
            tabm_cls.fit(X_train_df.astype(np.float32), pd.Series(y_train_cls, name='target'),
                         max_epochs=100, lr=0.01)
            y_pred_cls = tabm_cls.predict(X_test_df.astype(np.float32))
            y_prob_raw = tabm_cls.predict_proba(X_test_df.astype(np.float32))
            # Handle TabM returning 1-column or 2-column predict_proba
            if y_prob_raw.ndim == 1 or y_prob_raw.shape[1] == 1:
                y_prob_cls = y_prob_raw.ravel()
            else:
                y_prob_cls = y_prob_raw[:, 1]
            
            joblib.dump(tabm_reg, os.path.join(MODEL_DIR, f'{model_prefix}_tabm_reg.joblib'))
            joblib.dump(tabm_cls, os.path.join(MODEL_DIR, f'{model_prefix}_tabm_cls.joblib'))
            np.savez(os.path.join(MODEL_DIR, f'{model_prefix}_tabm_preds.npz'),
                     y_pred_reg=y_pred_reg, y_pred_cls=y_pred_cls, y_prob_cls=y_prob_cls)
            
            from src.evaluator import evaluate_regressor, evaluate_classifier
            tabm_metrics = {
                'reg': evaluate_regressor(y_test_reg_raw, y_pred_reg),
                'cls': evaluate_classifier(y_test_cls, y_pred_cls, y_prob_cls),
            }
            mark_model_done('tabm', model_prefix, tabm_metrics)
            results['tabm'] = {'y_pred_reg': y_pred_reg, 'y_pred_cls': y_pred_cls, 'y_prob_cls': y_prob_cls}
            logger.info(f"✅ TabM complete: R²={tabm_metrics['reg']['R²']:.4f}, "
                        f"F1={tabm_metrics['cls']['F1']:.4f}")
        except ImportError:
            logger.warning("TabM not installed. Skipping. Install: pip install deeptab")
        except Exception as e:
            logger.error(f"❌ TabM FAILED: {e}")
            logger.error(traceback.format_exc())
    else:
        # Load saved predictions from disk
        try:
            preds = np.load(os.path.join(MODEL_DIR, f'{model_prefix}_tabm_preds.npz'))
            results['tabm'] = {
                'y_pred_reg': preds['y_pred_reg'],
                'y_pred_cls': preds['y_pred_cls'],
                'y_prob_cls': preds['y_prob_cls'],
            }
            logger.info(f"⏭️  LOADED TabM ({model_prefix}) predictions from disk")
        except FileNotFoundError:
            logger.warning(f"TabM predictions file missing for {model_prefix}, skipping load")
    
    # ── Stacking Ensemble ────────────────────────────────
    if not is_model_done('stacked', model_prefix):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 Training: Stacked Ensemble ({model_prefix})")
            logger.info(f"{'='*60}")
            
            stack_reg = StackingRegressor(
                estimators=[
                    ('xgb', _create_model('xgboost', 'reg')),
                    ('lgb', _create_model('lightgbm', 'reg')),
                ],
                final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1,
            )
            stack_reg.fit(X_train, y_train_reg)
            y_pred_reg_log = stack_reg.predict(X_test)
            y_pred_reg = np.maximum(np.expm1(y_pred_reg_log), 0)
            
            stack_cls = StackingClassifier(
                estimators=[
                    ('xgb', _create_model('xgboost', 'cls', class_weight_ratio=class_weight_ratio)),
                    ('lgb', _create_model('lightgbm', 'cls', class_weight_ratio=class_weight_ratio)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
                cv=3, n_jobs=1,
            )
            stack_cls.fit(X_train, y_train_cls)
            y_pred_cls = stack_cls.predict(X_test)
            y_prob_cls = stack_cls.predict_proba(X_test)[:, 1]
            
            joblib.dump(stack_reg, os.path.join(MODEL_DIR, f'{model_prefix}_stacked_reg.joblib'))
            joblib.dump(stack_cls, os.path.join(MODEL_DIR, f'{model_prefix}_stacked_cls.joblib'))
            np.savez(os.path.join(MODEL_DIR, f'{model_prefix}_stacked_preds.npz'),
                     y_pred_reg=y_pred_reg, y_pred_cls=y_pred_cls, y_prob_cls=y_prob_cls)
            
            from src.evaluator import evaluate_regressor, evaluate_classifier
            stack_metrics = {
                'reg': evaluate_regressor(y_test_reg_raw, y_pred_reg),
                'cls': evaluate_classifier(y_test_cls, y_pred_cls, y_prob_cls),
            }
            mark_model_done('stacked', model_prefix, stack_metrics)
            results['stacked'] = {'y_pred_reg': y_pred_reg, 'y_pred_cls': y_pred_cls, 'y_prob_cls': y_prob_cls}
            logger.info(f"✅ Stacked Ensemble complete: R²={stack_metrics['reg']['R²']:.4f}, "
                        f"F1={stack_metrics['cls']['F1']:.4f}")
        except Exception as e:
            logger.error(f"❌ Stacked Ensemble FAILED: {e}")
            logger.error(traceback.format_exc())
    else:
        # Load saved predictions from disk
        try:
            preds = np.load(os.path.join(MODEL_DIR, f'{model_prefix}_stacked_preds.npz'))
            results['stacked'] = {
                'y_pred_reg': preds['y_pred_reg'],
                'y_pred_cls': preds['y_pred_cls'],
                'y_prob_cls': preds['y_prob_cls'],
            }
            logger.info(f"⏭️  LOADED Stacked Ensemble ({model_prefix}) predictions from disk")
        except FileNotFoundError:
            logger.warning(f"Stacked predictions file missing for {model_prefix}, skipping load")
    
    # Save feature columns
    with open(os.path.join(MODEL_DIR, f'{model_prefix}_features.json'), 'w') as f:
        json.dump(feature_cols, f)
    
    return results, y_test_reg_raw, y_test_cls
