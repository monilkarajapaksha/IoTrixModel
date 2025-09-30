"""
Alternative Implementation: Using RNN/GRU for IoT Prediction
This shows when and how RNN/GRU could be used for sequential predictions
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class IoT_RNN_Predictor:
    """RNN/GRU based IoT device prediction model"""
    
    def __init__(self, sequence_length=24):  # 24 hours lookback
        self.sequence_length = sequence_length
        self.models = {}
        self.scalers = {}
        self.device_names = [
            'TV', 'Washing Machine', 'AC', 'Refrigerator', 'Microwave',
            'Water Heater', 'Lights', 'Fan', 'Dishwasher', 'Electric Stove'
        ]
    
    def prepare_sequential_data(self, df, device_name):
        """Prepare sequential data for RNN training"""
        # Filter data for specific device
        device_data = df[df['device_name'] == device_name].copy()
        device_data = device_data.sort_values('timestamp')
        
        # Create features
        features = ['hour', 'minute', 'day_of_week', 'month', 'temperature']
        feature_data = device_data[features].values
        target_data = device_data['state'].values
        
        # Scale features
        scaler = MinMaxScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(feature_data_scaled)):
            # Use past 24 hours to predict current state
            X.append(feature_data_scaled[i-self.sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y), scaler
    
    def build_gru_model(self, input_shape):
        """Build GRU model architecture"""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, df, model_type='GRU'):
        """Train RNN models for each device"""
        print(f"Training {model_type} models...")
        
        for device in self.device_names:
            print(f"Training {model_type} for {device}...")
            
            # Prepare sequential data
            X, y, scaler = self.prepare_sequential_data(df, device)
            
            if len(X) < 100:  # Need sufficient data
                print(f"Insufficient data for {device}")
                continue
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build model
            if model_type == 'GRU':
                model = self.build_gru_model((X.shape[1], X.shape[2]))
            else:
                model = self.build_lstm_model((X.shape[1], X.shape[2]))
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"  {device} - Accuracy: {accuracy:.3f}")
            
            # Store model and scaler
            self.models[device] = model
            self.scalers[device] = scaler
    
    def predict_next_state(self, device_name, recent_sequence):
        """Predict next state based on recent sequence"""
        if device_name not in self.models:
            return 0, 0.5
        
        model = self.models[device_name]
        scaler = self.scalers[device_name]
        
        # Scale input sequence
        sequence_scaled = scaler.transform(recent_sequence)
        sequence_reshaped = sequence_scaled.reshape(1, self.sequence_length, -1)
        
        # Predict
        prediction_prob = model.predict(sequence_reshaped, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return prediction, float(prediction_prob)

# Comparison Analysis
class ModelComparison:
    """Compare Random Forest vs RNN/GRU approaches"""
    
    @staticmethod
    def analyze_approaches():
        """Analyze when to use each approach"""
        
        comparison = {
            "Random Forest": {
                "Best For": [
                    "Independent predictions (what state at specific time?)",
                    "Tabular/structured data",
                    "Feature-based patterns",
                    "Fast training and inference",
                    "Interpretable results"
                ],
                "Pros": [
                    "No sequential data requirement",
                    "Handles missing data well",
                    "Feature importance analysis",
                    "Robust to overfitting",
                    "Fast predictions",
                    "Works with small datasets"
                ],
                "Cons": [
                    "Cannot model temporal dependencies",
                    "No memory of past states",
                    "Limited sequential reasoning"
                ],
                "Use Case": "What should device state be at 7:30 PM today?"
            },
            
            "RNN/GRU": {
                "Best For": [
                    "Sequential predictions (what happens next?)",
                    "Time series forecasting",
                    "Temporal dependencies",
                    "Pattern continuation",
                    "Complex temporal relationships"
                ],
                "Pros": [
                    "Models temporal dependencies",
                    "Learns from sequences",
                    "Can predict multiple future steps",
                    "Captures long-term patterns",
                    "Good for time series"
                ],
                "Cons": [
                    "Requires sequential data",
                    "Slower training/inference",
                    "More complex architecture",
                    "Needs larger datasets",
                    "Harder to interpret",
                    "Gradient vanishing issues"
                ],
                "Use Case": "Given device states for last 24 hours, what will happen in next hour?"
            }
        }
        
        return comparison

# When RNN/GRU Would Be Better
def rnn_better_scenarios():
    """Scenarios where RNN/GRU would outperform Random Forest"""
    
    scenarios = {
        "1. Sequential Prediction": {
            "problem": "Predict device states for next 6 hours",
            "why_rnn": "Can model how one hour affects the next",
            "example": "If TV was ON for 3 hours, likely to turn OFF soon"
        },
        
        "2. Anomaly Detection": {
            "problem": "Detect unusual usage patterns",
            "why_rnn": "Learns normal sequences, flags deviations",
            "example": "AC running at 3 AM in winter is unusual"
        },
        
        "3. Energy Load Forecasting": {
            "problem": "Predict total power consumption over time",
            "why_rnn": "Models cumulative effects and trends",
            "example": "How total home energy usage changes throughout day"
        },
        
        "4. Adaptive Learning": {
            "problem": "Learn from recent usage changes",
            "why_rnn": "Can adapt to new patterns quickly",
            "example": "User changed routine, model adjusts predictions"
        },
        
        "5. Multi-Device Dependencies": {
            "problem": "Model how devices affect each other over time",
            "why_rnn": "Can learn cross-device temporal relationships",
            "example": "When AC turns ON, Fan usually turns OFF after 30 minutes"
        }
    }
    
    return scenarios

if __name__ == "__main__":
    # Analysis
    print("🤖 MODEL COMPARISON ANALYSIS")
    print("=" * 50)
    
    comparison = ModelComparison.analyze_approaches()
    
    for model_name, details in comparison.items():
        print(f"\n{model_name}:")
        print(f"Best For: {', '.join(details['Best For'])}")
        print(f"Use Case: {details['Use Case']}")
    
    print("\n🎯 WHEN TO USE RNN/GRU:")
    print("=" * 50)
    
    scenarios = rnn_better_scenarios()
    for scenario, details in scenarios.items():
        print(f"\n{scenario}")
        print(f"Problem: {details['problem']}")
        print(f"Why RNN: {details['why_rnn']}")
        print(f"Example: {details['example']}")
    
    print("\n💡 CONCLUSION FOR CURRENT PROJECT:")
    print("=" * 50)
    print("Random Forest is optimal because:")
    print("• We predict single time point (19:30 PM)")
    print("• No need for sequence prediction")
    print("• Pattern-based (not temporal dependency)")
    print("• Fast, interpretable, and accurate")
    print("\nRNN/GRU would be overkill for this specific use case!")