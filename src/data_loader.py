"""
Data Loader for Theme 1 (Police Violations) and Theme 2 (Astram Events).
"""
import pandas as pd
import json
from src.config import THEME1_FILE, THEME2_FILE, VIOLATION_SEVERITY, logger

def load_theme1():
    logger.info(f"Loading Theme 1 data from {THEME1_FILE}")
    df = pd.read_csv(THEME1_FILE, low_memory=False)
    
    # Filter out records without coordinates or dates
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime'])
    
    # Standardize validation_status
    df['validation_status'] = df['validation_status'].fillna('unknown').str.strip().str.lower()
    
    df_approved = df[df['validation_status'] == 'approved'].copy()
    df_rejected = df[df['validation_status'] == 'rejected'].copy()
    
    # Calculate severity weights for approved
    def calc_severity(violation_type_str):
        if pd.isna(violation_type_str):
            return 1
        try:
            # Check if it's a valid JSON string list
            v_types = json.loads(violation_type_str)
            if not isinstance(v_types, list):
                v_types = [v_types]
        except Exception:
            # Fallback if not standard JSON (e.g. malformed or flat string)
            clean_str = str(violation_type_str).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            v_types = [v.strip() for v in clean_str.split(',') if v.strip()]
            
        weights = [VIOLATION_SEVERITY.get(str(v).strip().upper(), 1) for v in v_types]
        return max(weights) if weights else 1

    df_approved['severity_weight'] = df_approved['violation_type'].apply(calc_severity)
    
    logger.info(f"Loaded Theme 1: {len(df_approved)} approved, {len(df_rejected)} rejected violations")
    return df_approved, df_rejected

def load_theme2():
    logger.info(f"Loading Theme 2 data from {THEME2_FILE}")
    df = pd.read_csv(THEME2_FILE)
    df = df.dropna(subset=['latitude', 'longitude', 'start_datetime'])
    logger.info(f"Loaded Theme 2: {len(df)} events")
    return df
