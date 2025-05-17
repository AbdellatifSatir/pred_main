import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Maintenance Monitor",
    page_icon="🔧"
)

# Load models and data
@st.cache_resource
def load_models():
    isolation_forest = joblib.load('isolation_forest_model.pkl')
    random_forest = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return isolation_forest, random_forest, scaler

@st.cache_data
def load_data():
    data = pd.read_csv('test_sample.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

# Prepare data for prediction
def prepare_features(df):
    # Add time features
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
    
    # Calculate rolling features for each machine
    for machine_id in df['machine_id'].unique():
        machine_data = df[df['machine_id'] == machine_id].sort_values('timestamp')
        
        for sensor in ['temperature', 'pressure', 'vibration', 'rotation_speed', 'voltage']:
            # 24-hour moving average
            df.loc[df['machine_id'] == machine_id, f'{sensor}_24h_avg'] = machine_data[sensor].rolling(24).mean().values
            
            # 24-hour moving standard deviation
            df.loc[df['machine_id'] == machine_id, f'{sensor}_24h_std'] = machine_data[sensor].rolling(24).std().values
            
            # Rate of change (hourly)
            df.loc[df['machine_id'] == machine_id, f'{sensor}_rate'] = machine_data[sensor].diff().values
    
    # Fill missing values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

# Predict anomalies
def predict_anomalies(df, scaler, isolation_forest, random_forest):
    # Features for prediction
    features = ['temperature', 'pressure', 'vibration', 'rotation_speed', 'voltage',
                'hour', 'day_of_week', 'is_weekend',
                'temperature_24h_avg', 'temperature_24h_std', 'temperature_rate',
                'pressure_24h_avg', 'pressure_24h_std', 'pressure_rate',
                'vibration_24h_avg', 'vibration_24h_std', 'vibration_rate',
                'rotation_speed_24h_avg', 'rotation_speed_24h_std', 'rotation_speed_rate',
                'voltage_24h_avg', 'voltage_24h_std', 'voltage_rate']
    
    # Make sure all features exist
    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features and scale
    X = df[features]
    X_scaled = scaler.transform(X)
    
    # Predict using both models
    df['if_anomaly'] = isolation_forest.predict(X_scaled)
    df['if_anomaly'] = np.where(df['if_anomaly'] == -1, 1, 0)
    
    df['rf_anomaly'] = random_forest.predict(X_scaled)
    
    # Combined prediction (if either model detects an anomaly)
    df['anomaly'] = np.logical_or(df['if_anomaly'], df['rf_anomaly']).astype(int)
    
    return df

# Simple plot function
def plot_machine_data(df, machine_id):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Filter data for the selected machine
    machine_data = df[df['machine_id'] == machine_id]
    
    # Plot temperature
    axs[0].plot(machine_data['timestamp'], machine_data['temperature'], 'b-', label='Temperature')
    axs[0].set_ylabel('Temperature')
    axs[0].grid(True)
    
    # Highlight anomalies
    anomalies = machine_data[machine_data['anomaly'] == 1]
    if len(anomalies) > 0:
        axs[0].scatter(anomalies['timestamp'], anomalies['temperature'], 
                     color='red', marker='*', s=100, label='Anomaly')
    
    axs[0].legend()
    
    # Plot vibration
    axs[1].plot(machine_data['timestamp'], machine_data['vibration'], 'g-', label='Vibration')
    axs[1].set_ylabel('Vibration')
    axs[1].grid(True)
    
    if len(anomalies) > 0:
        axs[1].scatter(anomalies['timestamp'], anomalies['vibration'], 
                     color='red', marker='*', s=100, label='Anomaly')
    
    axs[1].legend()
    
    # Plot pressure
    axs[2].plot(machine_data['timestamp'], machine_data['pressure'], 'orange', label='Pressure')
    axs[2].set_ylabel('Pressure')
    axs[2].set_xlabel('Time')
    axs[2].grid(True)
    
    if len(anomalies) > 0:
        axs[2].scatter(anomalies['timestamp'], anomalies['pressure'], 
                     color='red', marker='*', s=100, label='Anomaly')
    
    axs[2].legend()
    
    plt.tight_layout()
    return fig

# Main app
st.title("🔧 Machine Maintenance Monitor")

# Load data and models
try:
    isolation_forest, random_forest, scaler = load_models()
    data = load_data()
except Exception as e:
    st.error(f"Error loading data or models: {e}")
    st.stop()

# Simple UI with just machine selection
machine_ids = sorted(data['machine_id'].unique())
selected_machine = st.selectbox("Select Machine", machine_ids)

# Process data
prepared_data = prepare_features(data)
results = predict_anomalies(prepared_data, scaler, isolation_forest, random_forest)

# Filter for selected machine
machine_data = results[results['machine_id'] == selected_machine]

# Basic stats
total_anomalies = machine_data['anomaly'].sum()
anomaly_percentage = (total_anomalies / len(machine_data)) * 100

# Determine status
if anomaly_percentage > 10:
    status = "Critical"
    status_color = "red"
    recommendation = "Schedule maintenance immediately"
elif anomaly_percentage > 5:
    status = "Warning"
    status_color = "orange"
    recommendation = "Plan maintenance soon"
else:
    status = "Normal"
    status_color = "green"
    recommendation = "No maintenance needed"

# Display simple dashboard
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### Machine Status: <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
    st.write(f"**Recommendation:** {recommendation}")

with col2:
    st.metric("Anomalies Detected", f"{total_anomalies} ({anomaly_percentage:.1f}%)")

# Plot the data
st.subheader("Sensor Readings")
fig = plot_machine_data(machine_data, selected_machine)
st.pyplot(fig)

# Show recent anomalies
st.subheader("Recent Anomalies")
recent_anomalies = machine_data[machine_data['anomaly'] == 1].sort_values('timestamp', ascending=False).head(5)

if len(recent_anomalies) > 0:
    st.dataframe(recent_anomalies[['timestamp', 'temperature', 'vibration', 'pressure']])
else:
    st.info("No anomalies detected for this machine")

# Simple footer
st.markdown("---")
st.caption("Machine Learning Predictive Maintenance Demo")