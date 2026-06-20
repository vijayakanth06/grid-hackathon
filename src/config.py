"""
Central config that auto-detects environment (Linux GPU vs Windows local).
All paths, hyperparams, and GPU settings in one place.
"""
import os
import platform
import logging

# ─────────────────────────────────────────────────────────
# AUTO-DETECT ENVIRONMENT
# ─────────────────────────────────────────────────────────
IS_LINUX = platform.system() == 'Linux'
IS_WINDOWS = platform.system() == 'Windows'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if IS_LINUX:
    # Auto-detect GPU
    try:
        import torch
        HAS_GPU = torch.cuda.is_available()
        GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else 'None'
        GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if HAS_GPU else 0
    except ImportError:
        HAS_GPU = False
        GPU_NAME = 'None'
        GPU_MEM_GB = 0
else:
    HAS_GPU = False
    GPU_NAME = 'None'
    GPU_MEM_GB = 0

DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
HPO_DIR = os.path.join(BASE_DIR, 'hpo_results')

# Create directories
for d in [MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, HPO_DIR]:
    os.makedirs(d, exist_ok=True)

# File paths
THEME1_FILE = os.path.join(DATASET_DIR, 
    'jan to may police violation_anonymized791b166.csv')
THEME2_FILE = os.path.join(DATASET_DIR, 
    'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv')

# ─────────────────────────────────────────────────────────
# SPATIAL & TEMPORAL CONSTANTS
# ─────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 12.7, 13.4
LON_MIN, LON_MAX = 77.3, 77.9
GRID_PRECISION = 3
IST_OFFSET_HOURS = 5.5
TRAIN_TEST_SPLIT_DATE = '2024-03-01'
HOTSPOT_PERCENTILE = 75
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────
# GPU-AWARE MODEL CONFIGS
# ─────────────────────────────────────────────────────────
# XGBoost: tree_method='gpu_hist' on GPU, 'hist' on CPU
XGB_TREE_METHOD = 'hist' if HAS_GPU else 'hist'
XGB_DEVICE = 'cuda' if HAS_GPU else 'cpu'

# LightGBM: device='gpu' or 'cpu'
LGBM_DEVICE = 'gpu' if HAS_GPU else 'cpu'

# CatBoost: task_type='GPU' or 'CPU'
CATBOOST_TASK = 'GPU' if HAS_GPU else 'CPU'

# TabM: uses PyTorch → auto-detects CUDA via torch.cuda.is_available()

# ─────────────────────────────────────────────────────────
# HPO SETTINGS
# ─────────────────────────────────────────────────────────
HPO_N_TRIALS = 50          # Optuna trials per model
HPO_TIMEOUT_SECONDS = None # 1 hour max per model HPO
HPO_CV_FOLDS = 3           # TimeSeriesSplit folds for HPO

# ─────────────────────────────────────────────────────────
# VIOLATION SEVERITY WEIGHTS
# ─────────────────────────────────────────────────────────
VIOLATION_SEVERITY = {
    'WRONG PARKING': 3, 'NO PARKING': 3,
    'PARKING IN A MAIN ROAD': 5, 'PARKING NEAR ROAD CROSSING': 5,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 4,
    'DOUBLE PARKING': 5, 'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 4,
    'PARKING ON FOOTPATH': 2, 'DEFECTIVE NUMBER PLATE': 1,
    'REFUSE TO GO FOR HIRE': 1,
}

# ─────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOG_DIR, 'pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger('parksense')

# Print environment summary on import
logger.info(f"Environment: {'Linux' if IS_LINUX else 'Windows'}")
logger.info(f"GPU: {GPU_NAME} ({GPU_MEM_GB:.1f} GB)" if HAS_GPU else "GPU: None (CPU mode)")
logger.info(f"Base directory: {BASE_DIR}")
