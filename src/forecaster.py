"""
Prophet Forecasting for Police Station Violations.
"""
import os
import joblib
import pandas as pd
from src.config import logger, MODEL_DIR

def forecast_station_violations(df_approved, station_name, forecast_days=30):
    """
    Fits a Prophet forecasting model for a specific police station to forecast daily violations.
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed. Skipping forecasting. Install with pip install prophet")
        return None
        
    logger.info(f"Generating {forecast_days}-day daily forecast for police station: {station_name}")
    
    # Filter to station
    df_station = df_approved[df_approved['police_station'].astype(str).str.upper() == station_name.upper()].copy()
    if len(df_station) == 0:
        logger.warning(f"No violation data for station: {station_name}")
        return None
        
    # Aggregate daily
    df_daily = df_station.groupby('date').size().reset_index(name='y')
    df_daily['ds'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily[['ds', 'y']].sort_values('ds')
    
    if len(df_daily) < 10:
        logger.warning(f"Too few historical days ({len(df_daily)}) for station: {station_name} forecasting")
        return None
        
    # Fit Prophet
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_daily)
    
    # Forecast
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f'prophet_forecast_{station_name.replace(" ", "_")}.joblib')
    joblib.dump(model, model_path)
    
    logger.info(f"Forecasting complete for {station_name}. Model saved to {model_path}")
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
