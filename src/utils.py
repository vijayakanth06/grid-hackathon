"""
Utility helpers for distance calculations and SHAP model explainability.
"""
import os
import numpy as np
from src.config import logger

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def explain_model_shap(model, X, feature_cols, save_dir):
    """
    Generates and saves SHAP summary information.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg') # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        logger.info("Computing SHAP values for model explainability")
        
        # Check model type for suitable explainer
        model_str = str(type(model)).lower()
        if 'xgboost' in model_str or 'lightgbm' in model_str or 'catboost' in model_str or 'randomforest' in model_str or 'extratrees' in model_str:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'shap_summary.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"SHAP summary plot saved to {plot_path}")
        return plot_path
    except Exception as e:
        logger.warning(f"Failed to generate SHAP explanation: {e}")
        return None
