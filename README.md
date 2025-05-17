Industrial Equipment Predictive Maintenance
A simple machine learning solution for anomaly detection in industrial equipment to predict maintenance needs before failures occur.

Overview
This project demonstrates a complete predictive maintenance workflow using synthetic sensor data and machine learning models. It consists of three main components:

Synthetic Data Generation: Creates realistic sensor data for multiple machines with different anomaly patterns
Predictive Model Building: Trains both supervised and unsupervised models to detect anomalies
Interactive Dashboard: Simple Streamlit application to monitor machine health and detect anomalies
Features
Multi-sensor monitoring (temperature, pressure, vibration, rotation speed, voltage)
Anomaly detection using Isolation Forest (unsupervised) and Random Forest (supervised)
Machine health status assessment with maintenance recommendations
Visualization of sensor readings with highlighted anomalies
List of recent anomalies for quick review
Installation
Prerequisites
Python 3.8+
pip (Python package installer)
Setup
Clone this repository:
bash
git clone https://github.com/yourusername/predictive-maintenance.git
cd predictive-maintenance
Install required packages:
bash
pip install pandas numpy scikit-learn matplotlib joblib streamlit
Usage
1. Generate Synthetic Dataset
bash
python synthetic_dataset.py
This creates:

maintenance_dataset.csv: Dataset with 5 machines, 5 sensors, and various anomaly patterns
dataset_visualization.png: Visualization of the generated data
2. Train Predictive Models
bash
python build_models.py
This creates:

isolation_forest_model.pkl: Unsupervised anomaly detection model
random_forest_model.pkl: Supervised anomaly detection model
scaler.pkl: Feature standardization transformer
test_sample.csv: Sample data for the application
3. Run the Streamlit Dashboard
bash
streamlit run simple_streamlit_app.py
This launches a web interface where you can:

Select a machine to monitor
View its current health status
See sensor readings with highlighted anomalies
Review recent anomalies detected
Project Structure
├── synthetic_dataset.py     # Script to generate synthetic data
├── build_models.py          # Script to train and save ML models
├── simple_streamlit_app.py  # Simple dashboard application
├── maintenance_dataset.csv  # Generated synthetic dataset
├── isolation_forest_model.pkl  # Saved unsupervised model
├── random_forest_model.pkl  # Saved supervised model
├── scaler.pkl               # Saved feature scaler
└── test_sample.csv          # Sample data for the app
How It Works
Synthetic Data Generation
Creates data for 5 machines over 90 days with hourly readings
Includes daily and weekly patterns in sensor readings
Each machine has different anomaly patterns:
Machine 1: Sudden temperature spike
Machine 2: Gradual increase in vibration
Machine 3: Periodic pressure drops
Machine 4: Correlated anomaly between rotation and voltage
Machine 5: Random short anomalies
Machine Learning Models
Isolation Forest (Unsupervised):
Detects anomalies by isolating outliers in the feature space
Works without requiring labeled training data
Random Forest Classifier (Supervised):
Uses labeled anomalies to learn specific patterns
Provides probability scores for anomaly detection
Feature Engineering
Time-based features (hour, day of week, weekend flag)
Rolling statistics (24-hour averages, standard deviations)
Rate of change calculations
Potential Extensions
Real-time data simulation
Email/SMS alerts for critical anomalies
Cost analysis of preventive vs. reactive maintenance
Integration with real sensor data sources
Advanced time series models (LSTM, Prophet)
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
This project was created as a demonstration for a predictive maintenance hackathon
Inspired by real-world industrial IoT applications
Note: This is a demonstration project using synthetic data. In a real-world implementation, you would replace the data generation with actual sensor data from your equipment.

