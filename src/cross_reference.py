"""
Cross-referencing Theme 1 Grid with Theme 2 Events.
"""
import pandas as pd
from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_PRECISION, IST_OFFSET_HOURS, logger

def enrich_with_theme2(df_grid, df_events):
    """
    Enriches the grid dataframe with static and dynamic event counts from Theme 2.
    """
    logger.info("Enriching Theme 1 grid with Theme 2 events data")
    
    # Filter coordinates
    df_events = df_events[(df_events['latitude'] >= LAT_MIN) & (df_events['latitude'] <= LAT_MAX) &
                          (df_events['longitude'] >= LON_MIN) & (df_events['longitude'] <= LON_MAX)].copy()
    
    # Create grid cell matching format
    df_events['grid_cell'] = (
        df_events['latitude'].round(GRID_PRECISION).astype(str) + '_' + 
        df_events['longitude'].round(GRID_PRECISION).astype(str)
    )
    
    # Parse event dates and convert to IST
    df_events['start_datetime'] = pd.to_datetime(df_events['start_datetime'], errors='coerce')
    df_events['start_ist'] = df_events['start_datetime'] + pd.to_timedelta(IST_OFFSET_HOURS, unit='h')
    
    # Fill missing resolved datetime with start_datetime + 4 hours
    df_events['resolved_datetime'] = pd.to_datetime(df_events['resolved_datetime'], errors='coerce')
    df_events['resolved_ist'] = df_events['resolved_datetime'] + pd.to_timedelta(IST_OFFSET_HOURS, unit='h')
    df_events['resolved_ist'] = df_events['resolved_ist'].fillna(df_events['start_ist'] + pd.to_timedelta(4, unit='h'))
    
    # Event starting date and time bin
    df_events['event_date'] = df_events['start_ist'].dt.date
    df_events['event_time_bin'] = df_events['start_ist'].dt.hour // 4
    
    # Static features: total historical event density in this cell
    event_static = df_events['grid_cell'].value_counts().reset_index()
    event_static.columns = ['grid_cell', 'total_historical_events']
    
    # Dynamic features: events happening on this date and time bin
    df_events['event_date'] = pd.to_datetime(df_events['event_date'])
    event_dynamic = df_events.groupby(['grid_cell', 'event_date', 'event_time_bin']).size().reset_index(name='active_event_count')
    event_dynamic.rename(columns={'event_date': 'date', 'event_time_bin': 'time_bin'}, inplace=True)
    
    # Merge onto main grid
    df_grid['date'] = pd.to_datetime(df_grid['date'])
    df_enriched = pd.merge(df_grid, event_static, on='grid_cell', how='left')
    df_enriched['total_historical_events'] = df_enriched['total_historical_events'].fillna(0).astype(int)
    
    df_enriched = pd.merge(df_enriched, event_dynamic, on=['grid_cell', 'date', 'time_bin'], how='left')
    df_enriched['active_event_count'] = df_enriched['active_event_count'].fillna(0).astype(int)
    
    # Aggregate specific event causes as spatial columns
    causes = ['vehicle breakdown', 'potholes', 'construction', 'water logging', 'accident']
    for cause in causes:
        cause_df = df_events[df_events['event_cause'].fillna('').str.lower().str.contains(cause)]
        cause_counts = cause_df['grid_cell'].value_counts().reset_index()
        cause_counts.columns = ['grid_cell', f'event_count_{cause.replace(" ", "_")}']
        df_enriched = pd.merge(df_enriched, cause_counts, on='grid_cell', how='left')
        df_enriched[f'event_count_{cause.replace(" ", "_")}'] = df_enriched[f'event_count_{cause.replace(" ", "_")}'].fillna(0).astype(int)
        
    return df_enriched
