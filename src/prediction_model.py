import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from typing import Dict, Tuple, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_models import IoTDataModel, ModelFeatures, TimeSlotPrediction

class IoTPredictionModel:
    """Machine Learning model for predicting IoT device states"""
    
    def __init__(self):
        self.models = {}  # One model per device
        self.label_encoders = {}
        self.feature_columns = ['hour', 'minute', 'day_of_week', 'month', 
                               'temperature', 'is_weekend', 'is_peak_hour']
        self.device_names = IoTDataModel.DEVICE_NAMES
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training/prediction"""
        # Create feature matrix
        feature_df = df[self.feature_columns].copy()
        
        # Normalize temperature (optional)
        feature_df['temperature_norm'] = (feature_df['temperature'] - feature_df['temperature'].mean()) / feature_df['temperature'].std()
        
        # Add time-based features
        feature_df['is_morning'] = (feature_df['hour'] >= 6) & (feature_df['hour'] <= 10)
        feature_df['is_afternoon'] = (feature_df['hour'] >= 12) & (feature_df['hour'] <= 17)
        feature_df['is_evening'] = (feature_df['hour'] >= 18) & (feature_df['hour'] <= 22)
        feature_df['is_night'] = (feature_df['hour'] >= 23) | (feature_df['hour'] <= 5)
        
        # Convert boolean columns to int
        bool_columns = ['is_weekend', 'is_peak_hour', 'is_morning', 'is_afternoon', 'is_evening', 'is_night']
        for col in bool_columns:
            feature_df[col] = feature_df[col].astype(int)
        
        return feature_df
    
    def train(self, data_model: IoTDataModel) -> Dict[str, float]:
        """Train the prediction model"""
        print("Preparing training data...")
        training_df = data_model.prepare_training_data()
        
        if len(training_df) == 0:
            raise ValueError("No training data available")
        
        # Prepare features
        X = self.prepare_features(training_df)
        
        # Train a separate model for each device
        accuracies = {}
        
        for device in self.device_names:
            print(f"Training model for {device}...")
            
            # Get data for this device
            device_data = training_df[training_df['device_name'] == device].copy()
            
            if len(device_data) == 0:
                print(f"No data found for {device}, skipping...")
                continue
            
            # Prepare target variable
            y = device_data['target']
            X_device = X.loc[device_data.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_device, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies[device] = accuracy
            
            # Store model
            self.models[device] = model
            
            print(f"  Accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        print(f"\nTraining completed for {len(self.models)} devices")
        return accuracies
    
    def predict_device_state(self, device_name: str, features: ModelFeatures) -> Tuple[int, float]:
        """Predict state for a single device"""
        if not self.is_trained or device_name not in self.models:
            # Fallback to rule-based prediction
            return self._rule_based_prediction(device_name, features)
        
        # Prepare feature vector
        feature_dict = features.to_dict()
        feature_dict['temperature_norm'] = (feature_dict['temperature'] - 22.0) / 5.0  # Approximate normalization
        
        # Add time-based features
        feature_dict['is_morning'] = int(6 <= feature_dict['hour'] <= 10)
        feature_dict['is_afternoon'] = int(12 <= feature_dict['hour'] <= 17)
        feature_dict['is_evening'] = int(18 <= feature_dict['hour'] <= 22)
        feature_dict['is_night'] = int(feature_dict['hour'] >= 23 or feature_dict['hour'] <= 5)
        
        # Create feature vector in correct order
        feature_vector = np.array([[
            feature_dict['hour'],
            feature_dict['minute'],
            feature_dict['day_of_week'],
            feature_dict['month'],
            feature_dict['temperature'],
            feature_dict['is_weekend'],
            feature_dict['is_peak_hour'],
            feature_dict['temperature_norm'],
            feature_dict['is_morning'],
            feature_dict['is_afternoon'],
            feature_dict['is_evening'],
            feature_dict['is_night']
        ]])
        
        model = self.models[device_name]
        prediction = model.predict(feature_vector)[0]
        confidence = max(model.predict_proba(feature_vector)[0])
        
        return int(prediction), float(confidence)
    
    def _rule_based_prediction(self, device_name: str, features: ModelFeatures) -> Tuple[int, float]:
        """Fallback rule-based prediction when ML model is not available"""
        hour = features.hour
        temperature = features.temperature
        is_weekend = features.is_weekend
        
        # Simple rule-based logic for each device
        rules = {
            'TV': (hour >= 18 and hour <= 23) or (is_weekend and hour >= 10),
            'Washing Machine': (hour >= 8 and hour <= 10) or (hour >= 18 and hour <= 20),
            'AC': (temperature > 25) or (hour >= 20 and hour <= 22),
            'Refrigerator': True,  # Almost always on
            'Microwave': hour in [7, 8, 12, 13, 19, 20],
            'Water Heater': hour in [6, 7, 8, 19, 20, 21],
            'Lights': hour >= 18 and hour <= 23,
            'Fan': (temperature > 26) or (hour >= 21 and hour <= 23),
            'Dishwasher': hour >= 20 and hour <= 22,
            'Electric Stove': hour in [7, 8, 12, 13, 19, 20]
        }
        
        prediction = 1 if rules.get(device_name, False) else 0
        confidence = 0.6  # Lower confidence for rule-based predictions
        
        return prediction, confidence
    
    def predict_all_devices(self, timestamp: datetime, temperature: float) -> TimeSlotPrediction:
        """Predict states for all devices at a given time"""
        # Extract features from timestamp
        data_model = IoTDataModel()
        features = data_model.extract_features_from_timestamp(timestamp, temperature)
        
        # Predict for each device
        device_states = {}
        confidence_scores = {}
        
        for device in self.device_names:
            state, confidence = self.predict_device_state(device, features)
            device_states[device] = state
            confidence_scores[device] = confidence
        
        return TimeSlotPrediction(
            timestamp=timestamp,
            time_str=timestamp.strftime('%H:%M'),
            device_states=device_states,
            temperature=temperature,
            confidence_scores=confidence_scores
        )
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'device_names': self.device_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.device_names = model_data['device_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, device_name: str) -> Dict[str, float]:
        """Get feature importance for a specific device"""
        if device_name not in self.models:
            return {}
        
        model = self.models[device_name]
        feature_names = (self.feature_columns + 
                        ['temperature_norm', 'is_morning', 'is_afternoon', 'is_evening', 'is_night'])
        
        importance_dict = {}
        for i, importance in enumerate(model.feature_importances_):
            importance_dict[feature_names[i]] = importance
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def evaluate_model(self, data_model: IoTDataModel) -> Dict[str, Dict]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        training_df = data_model.prepare_training_data()
        X = self.prepare_features(training_df)
        
        evaluation_results = {}
        
        for device in self.device_names:
            if device not in self.models:
                continue
            
            device_data = training_df[training_df['device_name'] == device].copy()
            y_true = device_data['target']
            X_device = X.loc[device_data.index]
            
            model = self.models[device]
            y_pred = model.predict(X_device)
            
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate precision, recall for each class
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
            
            evaluation_results[device] = {
                'accuracy': accuracy,
                'precision_off': precision[0] if len(precision) > 0 else 0,
                'precision_on': precision[1] if len(precision) > 1 else 0,
                'recall_off': recall[0] if len(recall) > 0 else 0,
                'recall_on': recall[1] if len(recall) > 1 else 0,
                'f1_off': f1[0] if len(f1) > 0 else 0,
                'f1_on': f1[1] if len(f1) > 1 else 0
            }
        
        return evaluation_results