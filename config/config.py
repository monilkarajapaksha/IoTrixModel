# Configuration file for IoTrix Model

# Device Configuration
DEVICES = [
    'TV',
    'Washing Machine',
    'AC',
    'Refrigerator',
    'Microwave',
    'Water Heater',
    'Lights',
    'Fan',
    'Dishwasher',
    'Electric Stove'
]

# Power consumption estimates (in watts)
DEVICE_POWER_CONSUMPTION = {
    'TV': 150,
    'Washing Machine': 500,
    'AC': 1500,
    'Refrigerator': 200,
    'Microwave': 1200,
    'Water Heater': 3000,
    'Lights': 60,
    'Fan': 75,
    'Dishwasher': 1800,
    'Electric Stove': 2000
}

# Priority devices (should not be automatically turned off)
PRIORITY_DEVICES = ['Refrigerator']

# Peak electricity usage hours
PEAK_HOURS = [7, 8, 18, 19, 20, 21, 22]

# Temperature thresholds
TEMPERATURE_THRESHOLDS = {
    'AC_ON_TEMP': 25.0,
    'FAN_ON_TEMP': 26.0,
    'HEATER_ON_TEMP': 15.0,
    'NORMAL_TEMP': 22.0
}

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'class_weight': 'balanced'
}

# Data generation configuration
DATA_CONFIG = {
    'default_days': 30,
    'sampling_interval_minutes': 30,
    'temperature_base': 22.0,
    'temperature_variation': 5.0,
    'noise_std': 1.0
}

# File paths
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'output_dir': 'output',
    'default_data_file': 'iot_data.csv',
    'default_model_file': 'iot_prediction_model.joblib',
    'predictions_file': 'predictions.json'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# API configuration (if implementing REST API in future)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False
}

# Energy optimization settings
OPTIMIZATION_CONFIG = {
    'confidence_threshold': 0.7,
    'non_essential_devices': ['TV', 'Fan', 'Lights'],
    'max_savings_target_percentage': 30.0
}