"""
Smoke Test for ParkSense AI.
Verifies library imports, CUDA/GPU detection, and tests dummy training on GPU for XGBoost, LightGBM, and CatBoost.
"""
import sys
import numpy as np

def run_smoke_test():
    print("=" * 60)
    print("PARKSENSE AI — ENVIRONMENT SMOKE TEST")
    print("=" * 60)
    
    # ── 1. Check Python Version ──────────────────────────────────────────
    print(f"Python version: {sys.version}")
    
    # ── 2. Check Library Imports ─────────────────────────────────────────
    libs = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn", "sklearn"),
        ("xgboost", "xgb"),
        ("lightgbm", "lgb"),
        ("catboost", "catboost"),
        ("optuna", "optuna"),
        ("prophet", "prophet"),
        ("shap", "shap"),
        ("streamlit", "st"),
        ("torch", "torch"),
        ("joblib", "joblib")
    ]
    
    print("\n[1/4] Checking module imports:")
    failed = False
    for lib_name, short in libs:
        try:
            __import__(lib_name)
            print(f"  ✅ {lib_name:<12} : Loaded successfully")
        except ImportError as e:
            print(f"  ❌ {lib_name:<12} : FAILED ({e})")
            failed = True
            
    if failed:
        print("\n❌ Smoke test failed: Some required libraries are missing.")
        sys.exit(1)
        
    # ── 3. Check PyTorch & GPU Detection ───────────────────────────────
    print("\n[2/4] Checking CUDA / GPU status:")
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"  PyTorch CUDA available : {cuda_available}")
    if cuda_available:
        print(f"  GPU Device Name        : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Device Count       : {torch.cuda.device_count()}")
    else:
        print("  ⚠️ No GPU detected by PyTorch (CPU fallback mode will be active)")

    # ── 4. Test Dummy training on Device ──────────────────────────────
    print("\n[3/4] Testing model training on detected devices:")
    
    # Create small dummy dataset
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Test XGBoost
    try:
        import xgboost as xgb
        device_param = 'cuda' if cuda_available else 'cpu'
        tree_param = 'hist'
        print(f"  Testing XGBoost (device={device_param})...", end="")
        model = xgb.XGBRegressor(tree_method=tree_param, device=device_param, n_estimators=5)
        model.fit(X, y)
        print(" Success! ✅")
    except Exception as e:
        print(f" FAILED! ❌ ({e})")
        
    # Test LightGBM
    try:
        import lightgbm as lgb
        device_param = 'gpu' if cuda_available else 'cpu'
        print(f"  Testing LightGBM (device={device_param})...", end="")
        model = lgb.LGBMRegressor(device=device_param, n_estimators=5, verbose=-1)
        model.fit(X, y)
        print(" Success! ✅")
    except Exception as e:
        print(f" FAILED! ❌ ({e})")
        
    # Test CatBoost
    try:
        import catboost as cb
        device_param = 'GPU' if cuda_available else 'CPU'
        print(f"  Testing CatBoost (task_type={device_param})...", end="")
        model = cb.CatBoostRegressor(task_type=device_param, iterations=5, verbose=0)
        model.fit(X, y)
        print(" Success! ✅")
    except Exception as e:
        print(f" FAILED! ❌ ({e})")

    # Test TabM (optional)
    try:
        from deeptab.models import TabMClassifier
        print("  Testing TabM (CPU/GPU)...", end="")
        # Simple fit
        model = TabMClassifier()
        model.fit(X.astype(np.float32), y, max_epochs=2, lr=0.01)
        print(" Success! ✅")
    except Exception as e:
        print(f" Skipped/Not available ⚠️ ({e})")

    # ── 5. Project Source Imports ───────────────────────────────────────
    print("\n[4/4] Verifying ParkSense source code compilation:")
    try:
        from src.config import logger
        from src.checkpoint import load_checkpoint
        from src.gpu_utils import create_xgb_params
        from src.data_loader import load_theme1
        from src.feature_engineering import create_grid_cells
        from src.cross_reference import enrich_with_theme2
        from src.model_trainer import train_all_models
        from src.evaluator import evaluate_regressor
        from src.forecaster import forecast_station_violations
        print("  ✅ All src/ sub-modules compiled and loaded successfully!")
    except Exception as e:
        print(f"  ❌ Failed to load local source files: {e}")
        sys.exit(1)
        
    print("\n" + "=" * 60)
    print("✅ ENVIRONMENT READY — SMOKE TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
