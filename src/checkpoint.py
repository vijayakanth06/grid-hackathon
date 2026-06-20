"""
Checkpoint system for long-running training.
If model X is already saved → skip it on re-run.
If the pipeline crashes at model 4 → re-run starts at model 4.
"""
import os
import json
import joblib
from datetime import datetime
from src.config import CHECKPOINT_DIR, MODEL_DIR, logger

CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'progress.json')


def load_checkpoint() -> dict:
    """Load existing checkpoint or create empty one."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}. Starting fresh.")
    return {'completed': {}, 'started_at': None, 'last_updated': None}


def save_checkpoint(progress: dict):
    """Save checkpoint to disk."""
    progress['last_updated'] = datetime.now().isoformat()
    if progress['started_at'] is None:
        progress['started_at'] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def is_model_done(model_name: str, model_prefix: str) -> bool:
    """
    Check if a model has already been trained and saved.
    Checks BOTH the checkpoint file AND the actual .joblib file on disk.
    """
    progress = load_checkpoint()
    key = f"{model_prefix}_{model_name}"
    
    # Check checkpoint
    if key in progress.get('completed', {}):
        # Also verify the file actually exists on disk
        reg_path = os.path.join(MODEL_DIR, f'{model_prefix}_{model_name}_reg.joblib')
        cls_path = os.path.join(MODEL_DIR, f'{model_prefix}_{model_name}_cls.joblib')
        if os.path.exists(reg_path) and os.path.exists(cls_path):
            return True
    return False


def mark_model_done(model_name: str, model_prefix: str, metrics: dict):
    """Mark a model as completed in the checkpoint."""
    progress = load_checkpoint()
    key = f"{model_prefix}_{model_name}"
    progress['completed'][key] = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
    }
    save_checkpoint(progress)
    logger.info(f"✅ Checkpoint saved: {key}")


def get_completed_models() -> list:
    """Return list of all completed model keys."""
    progress = load_checkpoint()
    return list(progress.get('completed', {}).keys())
