"""
ParkSense AI Streamlit Dashboard.
Visualizes spatiotemporal hotspots, model leaderboards, forecasted trends, patrol routes, and economic impact.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap

# Set page configurations
st.set_page_config(
    page_title="ParkSense AI - Spatiotemporal Parking & Incident Intelligence",
    layout="wide",
    page_icon="🅿️"
)

# Centralized Theme Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .main-header h1 {
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .main-header p {
        font-size: 1.1rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── DATA LOADING ─────────────────────────────────────────
@st.cache_data
def load_dashboard_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
        
    v_file = os.path.join(base_dir, "dataset", "jan to may police violation_anonymized791b166.csv")
    e_file = os.path.join(base_dir, "dataset", "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
    
    if not os.path.exists(v_file):
        st.error(f"Dataset not found at {v_file}. Please check configuration.")
        st.stop()
        
    # Efficient chunked loading for Streamlit memory limits
    chunks = []
    for chunk in pd.read_csv(v_file, chunksize=50000, low_memory=False):
        approved = chunk[chunk['validation_status'] == 'approved'].copy()
        chunks.append(approved)
    df_violations = pd.concat(chunks, ignore_index=True)
    
    # Process Datetimes
    df_violations['created_datetime'] = pd.to_datetime(df_violations['created_datetime'], errors='coerce')
    df_violations['datetime_ist'] = df_violations['created_datetime'] + pd.to_timedelta(5.5, unit='h')
    df_violations['date'] = df_violations['datetime_ist'].dt.date
    df_violations['hour'] = df_violations['datetime_ist'].dt.hour
    df_violations['dayofweek'] = df_violations['datetime_ist'].dt.dayofweek
    df_violations['time_bin'] = df_violations['hour'] // 4
    df_violations['grid_cell'] = df_violations['latitude'].round(3).astype(str) + '_' + df_violations['longitude'].round(3).astype(str)
    
    # Calculate severity weights
    VIOLATION_SEVERITY = {
        'WRONG PARKING': 3, 'NO PARKING': 3,
        'PARKING IN A MAIN ROAD': 5, 'PARKING NEAR ROAD CROSSING': 5,
        'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC': 4,
        'DOUBLE PARKING': 5, 'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE': 4,
        'PARKING ON FOOTPATH': 2, 'DEFECTIVE NUMBER PLATE': 1,
        'REFUSE TO GO FOR HIRE': 1,
    }
    
    def calc_severity(violation_type_str):
        if pd.isna(violation_type_str):
            return 1
        clean_str = str(violation_type_str).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
        v_types = [v.strip() for v in clean_str.split(',') if v.strip()]
        weights = [VIOLATION_SEVERITY.get(str(v).strip().upper(), 1) for v in v_types]
        return max(weights) if weights else 1

    df_violations['severity_weight'] = df_violations['violation_type'].apply(calc_severity)

    # Precompute distance to nearest junction
    junction_meta = df_violations.dropna(subset=['junction_name']).groupby('junction_name').agg(
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

    grid_meta = df_violations.groupby('grid_cell').agg(
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean')
    ).reset_index()
    
    min_dist_to_junc = []
    for _, r_grid in grid_meta.iterrows():
        dists = haversine_np(r_grid['lon'], r_grid['lat'], 
                             junction_meta['lon'].values, junction_meta['lat'].values)
        min_dist_to_junc.append(np.min(dists) if len(dists) > 0 else 999.0)
    grid_meta['dist_to_nearest_junction'] = min_dist_to_junc
    
    df_violations = pd.merge(df_violations, grid_meta[['grid_cell', 'dist_to_nearest_junction']], on='grid_cell', how='left')

    # Process Events
    df_events = pd.read_csv(e_file)
    df_events['start_datetime'] = pd.to_datetime(df_events['start_datetime'], errors='coerce')
    df_events['start_ist'] = df_events['start_datetime'] + pd.to_timedelta(5.5, unit='h')
    df_events['date'] = df_events['start_ist'].dt.date
    df_events['hour'] = df_events['start_ist'].dt.hour
    df_events['time_bin'] = df_events['hour'] // 4
    df_events['grid_cell'] = df_events['latitude'].round(3).astype(str) + '_' + df_events['longitude'].round(3).astype(str)
    
    return df_violations, df_events, base_dir

df_violations, df_events, BASE_DIR = load_dashboard_data()

# Header block
st.markdown("""
<div class="main-header">
    <h1>PARKSENSE AI DASHBOARD</h1>
    <p>Spatiotemporal Parking Violation Forecasting & Traffic Incident Response Optimization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - Key Controls & Config
st.sidebar.image("https://img.icons8.com/color/144/parking.png", width=90)
st.sidebar.markdown("### Dashboard Filters")
vehicle_types = ['ALL'] + sorted(df_violations['vehicle_type'].dropna().unique().tolist())
sel_vtype = st.sidebar.selectbox("Filter by Vehicle Type", vehicle_types)

dow_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
sel_dow = st.sidebar.multiselect("Filter by Day of Week", list(dow_map.values()), default=list(dow_map.values()))

# Apply filters to violations
df_filtered = df_violations.copy()
if sel_vtype != 'ALL':
    df_filtered = df_filtered[df_filtered['vehicle_type'] == sel_vtype]
    
inverse_dow_map = {v: k for k, v in dow_map.items()}
selected_dow_ints = [inverse_dow_map[d] for d in sel_dow]
df_filtered = df_filtered[df_filtered['dayofweek'].isin(selected_dow_ints)]

# Top Level KPI Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <span class="metric-value">{len(df_filtered):,}</span><br>
        <span class="metric-label">Approved Violations</span>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <span class="metric-value">{df_filtered['grid_cell'].nunique():,}</span><br>
        <span class="metric-label">Active Hotspot Grid Cells</span>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <span class="metric-value">{len(df_events):,}</span><br>
        <span class="metric-label">Astram Traffic Incidents</span>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <span class="metric-value">V100 Server</span><br>
        <span class="metric-label">Active GPU Environment</span>
    </div>
    """, unsafe_allow_html=True)

# Create 5 Tab layout
tab_map, tab_leader, tab_trends, tab_patrol, tab_impact = st.tabs([
    "📍 Violations Heatmap",
    "🏆 ML Leaderboards",
    "📈 Spatiotemporal Forecaster",
    "🚓 Patrol Routing Optimizer",
    "💰 Economic & Congestion Impact"
])

# ─────────────────────────────────────────────────────────
# TAB 1: VIOLATIONS HEATMAP
# ─────────────────────────────────────────────────────────
with tab_map:
    st.markdown("### Interactive Geospatial Heatmap")
    st.markdown("Visualize spatiotemporal hotspots by density of violations, or view prioritized bottleneck areas using the **Congestion Impact Score**.")
    
    col_map_1, col_map_2 = st.columns([3, 1])
    with col_map_2:
        map_mode = st.radio("Map Visualization Mode", [
            "Raw Parking Violations Density",
            "Congestion Impact Score (Violations + Incidents + Junction Proximity)"
        ])
        
        hour_slider = st.slider("Filter by Hour of Day", 0, 23, (0, 23))
        df_map_subset = df_filtered[(df_filtered['hour'] >= hour_slider[0]) & (df_filtered['hour'] <= hour_slider[1])]
        df_events_subset = df_events[(df_events['hour'] >= hour_slider[0]) & (df_events['hour'] <= hour_slider[1])]
        
        st.markdown(f"**Map Observations Count:** {len(df_map_subset):,}")
        st.markdown("**Top Violation Locations:**")
        top_locs = df_map_subset['police_station'].value_counts().head(5)
        for loc, cnt in top_locs.items():
            st.write(f"- **{loc}**: {cnt:,} records")
            
    with col_map_1:
        # Folium map centered on Bengaluru
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles='CartoDB positron')
        
        if map_mode == "Raw Parking Violations Density":
            heat_data = df_map_subset[['latitude', 'longitude']].dropna().values.tolist()
            if heat_data:
                # Downsample for faster loading if too large
                if len(heat_data) > 15000:
                    indices = np.random.choice(len(heat_data), 15000, replace=False)
                    heat_data = [heat_data[i] for i in indices]
                HeatMap(heat_data, radius=10, max_zoom=13).add_to(m)
        else:
            # Congestion Impact Score Mode
            grid_viols = df_map_subset.groupby('grid_cell').agg(
                lat=('latitude', 'mean'),
                lon=('longitude', 'mean'),
                sev_sum=('severity_weight', 'sum'),
                dist_to_junc=('dist_to_nearest_junction', 'first')
            ).reset_index()
            
            grid_events = df_events_subset.groupby('grid_cell').size().reset_index(name='event_count')
            grid_merged = pd.merge(grid_viols, grid_events, on='grid_cell', how='left')
            grid_merged['event_count'] = grid_merged['event_count'].fillna(0)
            grid_merged['dist_to_junc'] = grid_merged['dist_to_junc'].fillna(1.0)
            
            # Compute score: Severity + (10 * active events) + (2 / (dist_to_junc + 0.05))
            grid_merged['congestion_impact'] = (
                grid_merged['sev_sum'] + 
                (10.0 * grid_merged['event_count']) + 
                (2.0 / (grid_merged['dist_to_junc'] + 0.05))
            )
            
            # Use score as weight
            heat_data = grid_merged[['lat', 'lon', 'congestion_impact']].dropna().values.tolist()
            if heat_data:
                # Normalize weights slightly for heatmap visual representation
                max_val = max([h[2] for h in heat_data]) if heat_data else 1
                heat_data = [[h[0], h[1], h[2]/max_val] for h in heat_data]
                HeatMap(heat_data, radius=15, max_zoom=13, blur=15).add_to(m)
            
        folium_static(m, width=900, height=550)

# ─────────────────────────────────────────────────────────
# TAB 2: ML LEADERBOARDS
# ─────────────────────────────────────────────────────────
with tab_leader:
    st.markdown("### Model Tournament Standings (Base vs Enhanced)")
    st.markdown("Compare model metrics trained using Theme 1 alone vs Theme 1 + Theme 2 cross-referenced features.")
    
    model_dir = os.path.join(BASE_DIR, 'models')
    
    # Read files
    r_base_file = os.path.join(model_dir, 'base_regression_leaderboard.csv')
    c_base_file = os.path.join(model_dir, 'base_classification_leaderboard.csv')
    r_enh_file = os.path.join(model_dir, 'enhanced_regression_leaderboard.csv')
    c_enh_file = os.path.join(model_dir, 'enhanced_classification_leaderboard.csv')
    
    has_results = os.path.exists(r_base_file)
    
    if not has_results:
        st.warning("⚠️ Training pipeline results not found. Please run `python run_pipeline.py` first.")
        st.stop()
    
    try:
        r_base = pd.read_csv(r_base_file)
        r_enh = pd.read_csv(r_enh_file)
        c_base = pd.read_csv(c_base_file)
        c_enh = pd.read_csv(c_enh_file)
        
        reg_data = pd.merge(r_base[['Model', 'R²', 'RMSE']], r_enh[['Model', 'R²', 'RMSE']], on='Model', suffixes=(' (Base)', ' (Enhanced)'))
        
        # Classification: include P@100 if available
        cls_base_cols = ['Model', 'F1', 'ROC_AUC']
        cls_enh_cols = ['Model', 'F1', 'ROC_AUC']
        if 'P@100' in c_base.columns:
            cls_base_cols.append('P@100')
            cls_enh_cols.append('P@100')
        cls_data = pd.merge(c_base[cls_base_cols], c_enh[cls_enh_cols], on='Model', suffixes=(' (Base)', ' (Enhanced)'))
    except Exception as e:
        st.error(f"Error loading leaderboard files: {e}")
        st.stop()
            
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.subheader("Regression Comparison ($R^2$ & RMSE)")
        st.dataframe(reg_data.style.background_gradient(cmap='Blues', subset=[c for c in reg_data.columns if 'R²' in c or 'RMSE' in c]))
        
        # Plot regression improvement
        fig = px.bar(reg_data, x='Model', y=[c for c in reg_data.columns if 'R²' in c], barmode='group',
                     title="Regression R² Score Comparison (Higher is Better)", labels={'value': 'R² Score', 'variable': 'Environment'})
        st.plotly_chart(fig, use_container_width=True)
        
    with col_l2:
        st.subheader("Classification Comparison (F1, ROC AUC & P@100)")
        st.dataframe(cls_data.style.background_gradient(cmap='Greens', subset=[c for c in cls_data.columns if 'F1' in c or 'ROC_AUC' in c or 'P@100' in c]))
        
        # Plot classification improvement
        fig = px.bar(cls_data, x='Model', y=[c for c in cls_data.columns if 'F1' in c], barmode='group',
                     title="Classification F1-Score Comparison (Higher is Better)", labels={'value': 'F1 Score', 'variable': 'Environment'})
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 3: SPATIOTEMPORAL FORECASTER
# ─────────────────────────────────────────────────────────
with tab_trends:
    st.markdown("### Daily Station Violation Forecaster (Prophet Model D)")
    st.markdown("Forecast upcoming daily violation counts for specific police station jurisdictions based on historical patterns.")
    
    selected_station = st.selectbox("Select Police Station", sorted(df_violations['police_station'].dropna().unique().tolist()))
    
    # Try to load prophet model
    model_path = os.path.join(BASE_DIR, 'models', f'prophet_forecast_{selected_station.replace(" ", "_")}.joblib')
    
    # Aggregate historical daily counts
    hist_daily = df_violations[df_violations['police_station'] == selected_station].groupby('date').size().reset_index(name='y')
    hist_daily['date'] = pd.to_datetime(hist_daily['date'])
    
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Plot actuals vs forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_daily['date'], y=hist_daily['y'], name='Historical Actuals', mode='markers+lines', line=dict(color='#1e3c72')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Prophet Forecast', line=dict(color='#ff7f0e', width=2)))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(255, 127, 14, 0.2)', showlegend=False))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(255, 127, 14, 0.2)', name='95% Confidence Interval'))
            
            fig.update_layout(title=f"30-day Violations Forecast: {selected_station}", xaxis_title="Date", yaxis_title="Daily Violations", template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to load/run forecasting model: {e}")
    else:
        st.info("ℹ️ Prophet model not found for this station. Showing 30-day forecast based on historical daily averages + weekly seasonality.")
        # Perform fallback forecasting logic
        hist_daily['dayofweek'] = hist_daily['date'].dt.dayofweek
        dow_avg = hist_daily.groupby('dayofweek')['y'].mean().to_dict()
        
        # Build 30 days future dates
        last_date = hist_daily['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
        future_y = [dow_avg.get(d.dayofweek, np.mean(list(dow_avg.values()))) for d in future_dates]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_daily['date'], y=hist_daily['y'], name='Historical Actuals', mode='lines', line=dict(color='#1e3c72')))
        fig.add_trace(go.Scatter(x=future_dates, y=future_y, name='Baseline Forecast (DOW average)', line=dict(color='#2ca02c', dash='dash', width=2)))
        
        fig.update_layout(title=f"Baseline Daily Violations Forecast: {selected_station}", xaxis_title="Date", yaxis_title="Daily Violations", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────
# TAB 4: PATROL ROUTING OPTIMIZER
# ─────────────────────────────────────────────────────────
with tab_patrol:
    st.markdown("### Tactical Patrol Route Optimization")
    st.markdown("Select a station and current time to calculate the optimal route connecting the top predicted violation hotspots, prioritized by **Congestion Impact**.")
    
    col_p1, col_p2 = st.columns([1, 2])
    
    with col_p1:
        patrol_station = st.selectbox("Patrol Station Jurisdiction", sorted(df_violations['police_station'].dropna().unique().tolist()), key='patrol_station')
        time_slot = st.selectbox("Time Bin (4-hour slot)", ["00:00 - 04:00 (Bin 0)", "04:00 - 08:00 (Bin 1)", "08:00 - 12:00 (Bin 2)", "12:00 - 16:00 (Bin 3)", "16:00 - 20:00 (Bin 4)", "20:00 - 00:00 (Bin 5)"])
        bin_idx = int(time_slot.split("Bin ")[1].replace(")", ""))
        
        # Get historical hotspots for that station & bin
        station_df = df_violations[(df_violations['police_station'] == patrol_station) & (df_violations['time_bin'] == bin_idx)].copy()
        
        if len(station_df) > 0:
            # Get Theme 2 events for matching grid cells and time bin
            station_grid_cells = station_df['grid_cell'].unique()
            station_events = df_events[(df_events['grid_cell'].isin(station_grid_cells)) & (df_events['time_bin'] == bin_idx)].copy()
            event_counts = station_events.groupby('grid_cell').size().to_dict()
            
            # Aggregate grid cells
            grid_summary = station_df.groupby('grid_cell').agg(
                lat=('latitude', 'mean'),
                lon=('longitude', 'mean'),
                count=('id', 'size'),
                sev_sum=('severity_weight', 'sum'),
                dist_to_junc=('dist_to_nearest_junction', 'first')
            ).reset_index()
            
            # Merge Theme 2 event counts
            grid_summary['event_count'] = grid_summary['grid_cell'].map(event_counts).fillna(0)
            grid_summary['dist_to_junc'] = grid_summary['dist_to_junc'].fillna(1.0)
            
            # Calculate Congestion Impact Score
            grid_summary['congestion_impact'] = (
                grid_summary['sev_sum'] + 
                (10.0 * grid_summary['event_count']) + 
                (2.0 / (grid_summary['dist_to_junc'] + 0.05))
            )
            
            # Get top 5 grid cells (hotspots) sorted by Congestion Impact Score
            top_hotspots = grid_summary.sort_values(by='congestion_impact', ascending=False).head(5)
            
            st.write(f"Found **{len(top_hotspots)}** key bottleneck zones in this jurisdiction.")
            st.dataframe(
                top_hotspots[['grid_cell', 'count', 'event_count', 'congestion_impact']]
                .rename(columns={
                    'count': 'Violations Count',
                    'event_count': 'Active Incidents',
                    'congestion_impact': 'Congestion Impact Score'
                })
            )
        else:
            top_hotspots = pd.DataFrame()
            st.info("No violation history for this time slot at this station.")
        
    with col_p2:
        if len(top_hotspots) > 0:
            # Center map on station average coords
            center_lat = top_hotspots['lat'].mean()
            center_lon = top_hotspots['lon'].mean()
            
            m_patrol = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
            
            # Simple route optimization (Nearest Neighbor from first point)
            coords = top_hotspots[['lat', 'lon']].values.tolist()
            route_idx = [0]
            unvisited = list(range(1, len(coords)))
            
            while unvisited:
                curr = coords[route_idx[-1]]
                next_idx = min(unvisited, key=lambda i: (coords[i][0]-curr[0])**2 + (coords[i][1]-curr[1])**2)
                route_idx.append(next_idx)
                unvisited.remove(next_idx)
                
            ordered_coords = [coords[i] for i in route_idx]
            
            # Draw line
            folium.PolyLine(ordered_coords, color="red", weight=4, opacity=0.8, tooltip="Optimized Route").add_to(m_patrol)
            
            # Add markers
            for step, idx in enumerate(route_idx):
                h = top_hotspots.iloc[idx]
                folium.Marker(
                    location=[h['lat'], h['lon']],
                    popup=f"Stop {step+1}: Grid {h['grid_cell']} (Impact: {h['congestion_impact']:.1f})",
                    icon=folium.Icon(color='red' if step == 0 else 'blue', icon='info-sign', prefix='fa')
                ).add_to(m_patrol)
                
            folium_static(m_patrol, width=800, height=450)
        else:
            st.info("No active route generated.")

# ─────────────────────────────────────────────────────────
# TAB 5: ECONOMIC & CONGESTION IMPACT
# ─────────────────────────────────────────────────────────
with tab_impact:
    st.markdown("### Economic & Environmental Congestion Impact")
    st.markdown("Reducing parking violations directly reduces carriageway bottlenecks. Calculate potential economic and fuel savings below.")
    
    target_reduction = st.slider("Target Violation Reduction (%)", 5, 50, 20, step=5)
    
    # Assumed baseline parameters based on traffic reports:
    # 1 violation delays 10 vehicles by an average of 5 minutes.
    # Fuel cost = Rs. 100/litre. Fuel waste rate = 0.5 litres per hour idling.
    # CO2 emission = 2.3 kg per litre fuel.
    
    total_violations_cnt = len(df_filtered)
    violations_reduced = int(total_violations_cnt * (target_reduction / 100.0))
    
    hours_saved = violations_reduced * 10 * 5 / 60.0 # 5 minutes per vehicle, 10 vehicles
    fuel_saved = hours_saved * 0.5 # 0.5L/hr
    money_saved = fuel_saved * 100 # Rs. 100/L
    co2_saved = fuel_saved * 2.3 # kg CO2
    
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("Total Congestion Hours Saved", f"{hours_saved:,.1f} hrs", delta=f"{violations_reduced:,} violations avoided")
    with c_m2:
        st.metric("Total Fuel Cost Saved (INR)", f"₹ {money_saved:,.1f}", delta=f"{fuel_saved:,.1f} litres saved")
    with c_m3:
        st.metric("Carbon Emissions Prevented", f"{co2_saved:,.1f} kg CO2", delta="Green Traffic Flow")
        
    # Plotly gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = money_saved,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Estimated Economic Return (₹ Saved / Day)", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [None, total_violations_cnt * 10 * 5 / 60.0 * 0.5 * 100]},
            'bar': {'color': "#1e3c72"},
            'steps': [
                {'range': [0, 100000], 'color': "lightgray"},
                {'range': [100000, 300000], 'color': "gray"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
