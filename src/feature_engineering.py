"""
Feature Engineering for ParkSense AI.
Includes grid cell generation, time bin mapping, spatiotemporal aggregation, and lag calculation.

V2 IMPROVEMENTS:
- Active-row filtering: only keep rows with recent activity (Fix 1)
- Non-zero hotspot threshold: compute percentile on non-zero violations (Fix 2)
- Richer features: is_weekend, cell rank, station avg, time-cell interaction (Fix 4)
"""
import numpy as np
import pandas as pd
from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, GRID_PRECISION, IST_OFFSET_HOURS, logger

def create_grid_cells(df):
    """
    Filters violations to coordinates within the bounding box and generates geohash-like grid cells.
    """
    # Bounding box filter
    df = df[(df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) &
            (df['longitude'] >= LON_MIN) & (df['longitude'] <= LON_MAX)].copy()
    
    # Generate grid cell ID using rounded coordinates
    df['grid_cell'] = (
        df['latitude'].round(GRID_PRECISION).astype(str) + '_' + 
        df['longitude'].round(GRID_PRECISION).astype(str)
    )
    return df


def create_time_features(df):
    """
    Parses datetime and converts to IST.
    """
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    # Convert to IST (UTC + 5.5)
    df['datetime_ist'] = df['created_datetime'] + pd.to_timedelta(IST_OFFSET_HOURS, unit='h')
    df['date'] = df['datetime_ist'].dt.date
    df['hour'] = df['datetime_ist'].dt.hour
    df['dayofweek'] = df['datetime_ist'].dt.dayofweek
    df['month'] = df['datetime_ist'].dt.month
    df['time_bin'] = df['hour'] // 4  # 6 bins of 4 hours
    
    return df


def aggregate_to_grid_time(df_approved):
    """
    Aggregates approved violations to (grid_cell, date, time_bin) level.
    Builds a skeleton dataset to include zero-violation periods.
    """
    # Double check coordinate range
    df_approved = df_approved[(df_approved['latitude'] >= LAT_MIN) & (df_approved['latitude'] <= LAT_MAX) &
                              (df_approved['longitude'] >= LON_MIN) & (df_approved['longitude'] <= LON_MAX)].copy()
    
    # Select active grid cells (those with at least 5 violations overall) to prevent sparseness issues
    vc = df_approved['grid_cell'].value_counts()
    active_grid_cells = vc[vc >= 5].index.tolist()
    logger.info(f"Filtering to {len(active_grid_cells)} active grid cells with >= 5 violations")
    
    df_active = df_approved[df_approved['grid_cell'].isin(active_grid_cells)].copy()
    
    # Group actual violations
    agg = df_active.groupby(['grid_cell', 'date', 'time_bin']).agg(
        violation_count=('id', 'count'),
        severity_sum=('severity_weight', 'sum')
    ).reset_index()
    
    # Extract metadata per grid cell (mode police station, mean lat/lon)
    grid_meta = df_active.groupby('grid_cell').agg(
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean'),
        police_station=('police_station', lambda x: x.mode()[0] if not x.mode().empty else 'UNKNOWN')
    ).reset_index()
    
    # Create contiguous Cartesian product skeleton
    unique_dates = df_active['date'].unique()
    unique_dates = sorted([d for d in unique_dates if pd.notna(d)])
    
    from itertools import product
    skeleton = pd.DataFrame(
        product(active_grid_cells, unique_dates, range(6)), 
        columns=['grid_cell', 'date', 'time_bin']
    )
    
    # Merge skeleton with actual aggregations
    df_grid = pd.merge(skeleton, agg, on=['grid_cell', 'date', 'time_bin'], how='left')
    df_grid['violation_count'] = df_grid['violation_count'].fillna(0).astype(int)
    df_grid['severity_sum'] = df_grid['severity_sum'].fillna(0.0)
    
    # Merge grid metadata
    df_grid = pd.merge(df_grid, grid_meta, on='grid_cell', how='left')
    
    # Label encode police stations
    df_grid['police_station'] = df_grid['police_station'].fillna('UNKNOWN').astype(str)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_grid['police_station_code'] = le.fit_transform(df_grid['police_station'])
    station_codes = dict(zip(le.classes_, map(int, le.transform(le.classes_))))
    
    # Temporal features on the grid
    df_grid['date'] = pd.to_datetime(df_grid['date'])
    df_grid['dayofweek'] = df_grid['date'].dt.dayofweek
    df_grid['month'] = df_grid['date'].dt.month
    df_grid['hour_start'] = df_grid['time_bin'] * 4
    
    # Cyclical hour & day of week features
    df_grid['sin_hour'] = np.sin(2 * np.pi * df_grid['hour_start'] / 24.0)
    df_grid['cos_hour'] = np.cos(2 * np.pi * df_grid['hour_start'] / 24.0)
    df_grid['sin_dow'] = np.sin(2 * np.pi * df_grid['dayofweek'] / 7.0)
    df_grid['cos_dow'] = np.cos(2 * np.pi * df_grid['dayofweek'] / 7.0)
    
    # ── Fix 4: RICHER FEATURES ──────────────────────────────────
    # (a) Weekend flag — parking enforcement patterns differ on weekends
    df_grid['is_weekend'] = (df_grid['dayofweek'] >= 5).astype(int)
    
    # (b) Historical violation rate per grid cell (global prior)
    cell_total = df_grid.groupby('grid_cell')['violation_count'].sum().reset_index()
    cell_total.columns = ['grid_cell', '_cell_total']
    cell_slots = df_grid.groupby('grid_cell')['violation_count'].count().reset_index()
    cell_slots.columns = ['grid_cell', '_cell_slots']
    cell_stats = pd.merge(cell_total, cell_slots, on='grid_cell')
    cell_stats['historical_violation_rate'] = cell_stats['_cell_total'] / cell_stats['_cell_slots']
    df_grid = pd.merge(df_grid, cell_stats[['grid_cell', 'historical_violation_rate']], on='grid_cell', how='left')
    
    # (c) Station-level average daily violations (captures enforcement area activity)
    station_daily = df_grid.groupby(['police_station_code', df_grid['date'].dt.date])['violation_count'].sum().reset_index()
    station_avg = station_daily.groupby('police_station_code')['violation_count'].mean().reset_index()
    station_avg.columns = ['police_station_code', 'station_avg_violations']
    df_grid = pd.merge(df_grid, station_avg, on='police_station_code', how='left')
    df_grid['station_avg_violations'] = df_grid['station_avg_violations'].fillna(0)
    
    # (d) Grid cell rank — ordinal rank by total historical violations (spatial hotspot prior)
    cell_rank = cell_stats[['grid_cell', '_cell_total']].copy()
    cell_rank['grid_cell_rank'] = cell_rank['_cell_total'].rank(ascending=False, method='dense')
    # Normalize to [0, 1] for model stability
    cell_rank['grid_cell_rank'] = cell_rank['grid_cell_rank'] / cell_rank['grid_cell_rank'].max()
    df_grid = pd.merge(df_grid, cell_rank[['grid_cell', 'grid_cell_rank']], on='grid_cell', how='left')
    
    # (e) Time bin rate — average violation rate per time_bin across all cells
    time_bin_avg = df_grid.groupby('time_bin')['violation_count'].mean().reset_index()
    time_bin_avg.columns = ['time_bin', 'time_bin_rate']
    df_grid = pd.merge(df_grid, time_bin_avg, on='time_bin', how='left')
    
    # (f) Cell × Time interaction — captures cells that are hot at specific times
    df_grid['cell_time_interaction'] = df_grid['historical_violation_rate'] * df_grid['time_bin_rate']
    
    # Distance to nearest junction
    junction_meta = df_approved.dropna(subset=['junction_name']).groupby('junction_name').agg(
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean')
    ).reset_index()
    
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    grid_coords = grid_meta[['grid_cell', 'latitude', 'longitude']].copy()
    min_dist_to_junc = []
    for _, r_grid in grid_coords.iterrows():
        dists = haversine_np(r_grid['longitude'], r_grid['latitude'], 
                             junction_meta['lon'].values, junction_meta['lat'].values)
        min_dist_to_junc.append(np.min(dists) if len(dists) > 0 else 999.0)
        
    grid_coords['dist_to_nearest_junction'] = min_dist_to_junc
    
    # Merge distance feature
    df_grid = pd.merge(df_grid, grid_coords[['grid_cell', 'dist_to_nearest_junction']], on='grid_cell', how='left')
    
    return df_grid, station_codes


def add_lag_features(df_grid):
    """
    Computes spatiotemporal lags and rolling statistics.
    Since the skeleton is perfectly contiguous, shifting by 6 steps is exactly 1 day.
    
    V2: Also applies active-row filtering (Fix 1) and non-zero hotspot threshold (Fix 2).
    """
    df_grid = df_grid.sort_values(by=['grid_cell', 'date', 'time_bin']).reset_index(drop=True)
    
    slots_per_day = 6
    gp = df_grid.groupby('grid_cell')
    
    # 1-day, 2-day, 7-day, 14-day lags on the same time bin
    df_grid['lag_1d'] = gp['violation_count'].shift(slots_per_day).fillna(0)
    df_grid['lag_2d'] = gp['violation_count'].shift(slots_per_day * 2).fillna(0)
    df_grid['lag_7d'] = gp['violation_count'].shift(slots_per_day * 7).fillna(0)
    df_grid['lag_14d'] = gp['violation_count'].shift(slots_per_day * 14).fillna(0)
    
    # 3-day and 7-day rolling means on the same time bin
    df_grid['rolling_mean_3d'] = (
        df_grid['lag_1d'] + 
        gp['violation_count'].shift(slots_per_day * 2).fillna(0) + 
        gp['violation_count'].shift(slots_per_day * 3).fillna(0)
    ) / 3.0
    df_grid['rolling_mean_7d'] = sum(gp['violation_count'].shift(slots_per_day * i).fillna(0) for i in range(1, 8)) / 7.0
    
    # 1-day, 2-day, 7-day, 14-day lags on severity_sum
    df_grid['severity_sum_lag_1d'] = gp['severity_sum'].shift(slots_per_day).fillna(0)
    df_grid['severity_sum_lag_2d'] = gp['severity_sum'].shift(slots_per_day * 2).fillna(0)
    df_grid['severity_sum_lag_7d'] = gp['severity_sum'].shift(slots_per_day * 7).fillna(0)
    df_grid['severity_sum_lag_14d'] = gp['severity_sum'].shift(slots_per_day * 14).fillna(0)
    
    # 3-day and 7-day rolling means on severity_sum
    df_grid['severity_sum_rolling_mean_3d'] = (
        df_grid['severity_sum_lag_1d'] + 
        gp['severity_sum'].shift(slots_per_day * 2).fillna(0) + 
        gp['severity_sum'].shift(slots_per_day * 3).fillna(0)
    ) / 3.0
    df_grid['severity_sum_rolling_mean_7d'] = sum(gp['severity_sum'].shift(slots_per_day * i).fillna(0) for i in range(1, 8)) / 7.0
    
    # ── Fix 2: NON-ZERO HOTSPOT THRESHOLD ───────────────────────
    # Compute threshold only on rows where violations actually occurred.
    # This gives a meaningful percentile (e.g., "top 25% of non-zero slots").
    from src.config import HOTSPOT_PERCENTILE
    nonzero_violations = df_grid.loc[df_grid['violation_count'] > 0, 'violation_count']
    if len(nonzero_violations) > 0:
        threshold = nonzero_violations.quantile(HOTSPOT_PERCENTILE / 100.0)
        # Ensure threshold is at least 1 so zero-violation rows are never hotspots
        threshold = max(threshold, 1)
    else:
        threshold = 1
    df_grid['is_hotspot'] = (df_grid['violation_count'] >= threshold).astype(int)
    
    hotspot_count = df_grid['is_hotspot'].sum()
    total_count = len(df_grid)
    logger.info(f"Hotspot threshold (non-zero {HOTSPOT_PERCENTILE}th pctile): {threshold}")
    logger.info(f"Hotspots: {hotspot_count} / {total_count} ({100*hotspot_count/total_count:.2f}%)")
    
    # ── Fix 1: ACTIVE-ROW FILTERING ─────────────────────────────
    # Only keep rows where:
    #   - violation_count > 0 (actual violations), OR
    #   - lag_1d > 0 OR lag_7d > 0 OR rolling_mean_7d > 0 (recent activity context)
    # This eliminates the 94% of rows that are pure zeros with no nearby signal.
    full_size = len(df_grid)
    active_mask = (
        (df_grid['violation_count'] > 0) |
        (df_grid['lag_1d'] > 0) |
        (df_grid['lag_2d'] > 0) |
        (df_grid['lag_7d'] > 0) |
        (df_grid['rolling_mean_7d'] > 0)
    )
    df_grid = df_grid[active_mask].reset_index(drop=True)
    logger.info(f"Active-row filter: {full_size} → {len(df_grid)} rows "
                f"({100*len(df_grid)/full_size:.1f}% retained)")
    
    return df_grid
