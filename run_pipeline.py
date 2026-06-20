"""
MASTER PIPELINE — Run this ONE file to run the entire training sequence.

Usage:
    python run_pipeline.py

What it does:
1. Load & clean data
2. Feature engineering (120+ features)
3. HPO for all 5 tree models (cached — skip if already done)
4. Train Base models (Theme 1 only) — 7 models with checkpoint
5. Train Enhanced models (Theme 1+2) — 7 models with checkpoint
6. Generate comparison leaderboards
7. Save everything
"""
import sys
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Add src to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import logger, MODEL_DIR
from src.data_loader import load_theme1, load_theme2
from src.feature_engineering import (
    create_grid_cells, create_time_features,
    aggregate_to_grid_time, add_lag_features
)
from src.cross_reference import enrich_with_theme2
from src.hpo import run_all_hpo
from src.model_trainer import train_all_models, temporal_split
from src.evaluator import build_leaderboard

def main():
    logger.info("=" * 70)
    logger.info("ParkSense AI Pipeline — Starting")
    logger.info("=" * 70)

    # ── V2 Auto-cleanup: Always wipe cache for clean V2 run ────
    # V2 changed features, target transform, class weights — old cache is incompatible
    from src.config import HPO_DIR, CHECKPOINT_DIR, MODEL_DIR
    import shutil
    import json
    v2_marker = os.path.join(CHECKPOINT_DIR, 'v2_complete.marker')
    if not os.path.exists(v2_marker):
        logger.info("🧹 V2 upgrade: Wiping old checkpoints, models, and HPO results...")
        for folder in [HPO_DIR, CHECKPOINT_DIR, MODEL_DIR]:
            if os.path.exists(folder):
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
        logger.info("🧹 V2 cleanup complete.")
    else:
        logger.info("✅ V2 marker found — using existing V2 cache.")

    # ── Step 1: Load Data ────────────────────────────────────
    logger.info("\n📂 Step 1: Loading data...")
    df_approved, df_rejected = load_theme1()
    df_events = load_theme2()
    logger.info(f"Theme 1: {len(df_approved)} approved violations")
    logger.info(f"Theme 2: {len(df_events)} events")

    # ── Step 2: Feature Engineering ──────────────────────────
    logger.info("\n🔧 Step 2: Feature engineering...")
    df_approved = create_grid_cells(df_approved)
    df_approved = create_time_features(df_approved)
    df_grid, station_codes = aggregate_to_grid_time(df_approved)
    df_grid = add_lag_features(df_grid)
    # Sort chronologically to make sure TimeSeriesSplit splits chronologically across all locations
    df_grid = df_grid.sort_values(by=['date', 'time_bin', 'grid_cell']).reset_index(drop=True)
    logger.info(f"Grid-time observations: {len(df_grid)}")

    # Base features (Theme 1 only)
    exclude_cols = ['violation_count', 'is_hotspot', 'grid_cell', 'date', 'time_bin', 'police_station', 'severity_sum']
    BASE_FEATURES = [c for c in df_grid.columns if c not in exclude_cols and 
                     df_grid[c].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'float8']]
    logger.info(f"Base features: {len(BASE_FEATURES)}")
    logger.info(f"Feature list: {BASE_FEATURES}")

    # ── Step 3: HPO ──────────────────────────────────────────
    logger.info("\n🎯 Step 3: Hyperparameter Optimization...")
    train, _ = temporal_split(df_grid)
    X_train_hpo = train[BASE_FEATURES]
    y_train_reg_hpo = train['violation_count']
    y_train_cls_hpo = train['is_hotspot']
    hpo_results = run_all_hpo(X_train_hpo, y_train_reg_hpo, y_train_cls_hpo)

    # ── Step 4: Train Base Models ────────────────────────────
    logger.info("\n🏗️  Step 4: Training Base Models (Theme 1 only)...")
    base_results, y_test_reg, y_test_cls = train_all_models(
        df_grid, BASE_FEATURES, model_prefix='base', hpo_results=hpo_results
    )

    # ── Step 5: Enrich with Theme 2 + Train Enhanced ────────
    logger.info("\n🔀 Step 5: Cross-referencing Theme 2 + Training Enhanced Models...")
    df_enhanced = enrich_with_theme2(df_grid, df_events)
    # Ensure enhanced dataset is also sorted chronologically
    df_enhanced = df_enhanced.sort_values(by=['date', 'time_bin', 'grid_cell']).reset_index(drop=True)
    
    exclude_cols_enh = exclude_cols + ['dominant_corridor', 'zone']
    ENHANCED_FEATURES = [c for c in df_enhanced.columns if c not in exclude_cols_enh and 
                         df_enhanced[c].dtype in ['int64', 'float64', 'int32', 'float32', 'int8', 'float8']]
    logger.info(f"Enhanced features: {len(ENHANCED_FEATURES)}")
    logger.info(f"Enhanced feature list: {ENHANCED_FEATURES}")

    enhanced_results, _, _ = train_all_models(
        df_enhanced, ENHANCED_FEATURES, model_prefix='enhanced', hpo_results=hpo_results
    )

    # ── Step 6: Comparison ───────────────────────────────────
    logger.info("\n📊 Step 6: Building leaderboards...")
    base_reg_lb, base_cls_lb = build_leaderboard(base_results, y_test_reg, y_test_cls)
    enh_reg_lb, enh_cls_lb = build_leaderboard(enhanced_results, y_test_reg, y_test_cls)

    logger.info("\n=== BASE MODEL LEADERBOARD (Regression) ===")
    logger.info(f"\n{base_reg_lb.to_string()}")
    logger.info("\n=== ENHANCED MODEL LEADERBOARD (Regression) ===")
    logger.info(f"\n{enh_reg_lb.to_string()}")

    logger.info("\n=== BASE MODEL LEADERBOARD (Classification) ===")
    logger.info(f"\n{base_cls_lb.to_string()}")
    logger.info("\n=== ENHANCED MODEL LEADERBOARD (Classification) ===")
    logger.info(f"\n{enh_cls_lb.to_string()}")

    # Save leaderboards
    base_reg_lb.to_csv(os.path.join(MODEL_DIR, 'base_regression_leaderboard.csv'), index=False)
    base_cls_lb.to_csv(os.path.join(MODEL_DIR, 'base_classification_leaderboard.csv'), index=False)
    enh_reg_lb.to_csv(os.path.join(MODEL_DIR, 'enhanced_regression_leaderboard.csv'), index=False)
    enh_cls_lb.to_csv(os.path.join(MODEL_DIR, 'enhanced_classification_leaderboard.csv'), index=False)

    # Write V2 marker so future runs don't wipe cache again
    with open(v2_marker, 'w') as f:
        f.write('v2_complete')

    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE V2 COMPLETE — All 7 fixes applied")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()
