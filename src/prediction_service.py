from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json

from data_models import IoTDataModel, TimeSlotPrediction, create_time_slot_prediction
from prediction_model import IoTPredictionModel

class IoTPredictionService:
    """Service for making IoT device state predictions"""
    
    def __init__(self):
        self.model = IoTPredictionModel()
        self.data_model = IoTDataModel()
        self.is_model_loaded = False
        
    def load_data(self, data_path: str) -> None:
        """Load training data from CSV file"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        self.data_model.load_from_dataframe(df)
        print(f"Loaded {len(df)} records")
        
    def train_model(self) -> Dict[str, float]:
        """Train the prediction model using loaded data"""
        if len(self.data_model.data) == 0:
            raise ValueError("No data loaded. Please load data first.")
        
        print("Training prediction model...")
        accuracies = self.model.train(self.data_model)
        self.is_model_loaded = True
        
        print("\nModel Training Results:")
        for device, accuracy in accuracies.items():
            print(f"  {device}: {accuracy:.3f}")
        
        return accuracies
        
    def load_trained_model(self, model_path: str) -> None:
        """Load a pre-trained model"""
        self.model.load_model(model_path)
        self.is_model_loaded = True
        
    def save_model(self, model_path: str) -> None:
        """Save the trained model"""
        if not self.is_model_loaded:
            raise ValueError("No model to save. Train or load a model first.")
        self.model.save_model(model_path)
        
    def predict_for_time(self, time_str: str, date_str: Optional[str] = None, 
                        temperature: float = 22.0) -> TimeSlotPrediction:
        """
        Predict device states for a specific time
        
        Args:
            time_str: Time in format "HH:MM" (e.g., "19:30" for 7:30 PM)
            date_str: Date in format "YYYY-MM-DD" (optional, defaults to today)
            temperature: Temperature in Celsius
            
        Returns:
            TimeSlotPrediction object with device states
        """
        if not self.is_model_loaded:
            print("Warning: No trained model loaded. Using rule-based predictions.")
        
        # Parse time
        try:
            hour, minute = map(int, time_str.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Invalid time format")
        except (ValueError, AttributeError):
            raise ValueError("Time must be in format 'HH:MM' (e.g., '19:30')")
        
        # Create timestamp
        if date_str:
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Date must be in format 'YYYY-MM-DD'")
            timestamp = datetime.combine(date, datetime.min.time().replace(hour=hour, minute=minute))
        else:
            timestamp = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # Make prediction
        prediction = self.model.predict_all_devices(timestamp, temperature)
        
        return prediction
        
    def predict_for_multiple_times(self, time_slots: List[str], 
                                 date_str: Optional[str] = None,
                                 temperature: float = 22.0) -> List[TimeSlotPrediction]:
        """Predict device states for multiple time slots"""
        predictions = []
        
        for time_str in time_slots:
            try:
                prediction = self.predict_for_time(time_str, date_str, temperature)
                predictions.append(prediction)
            except ValueError as e:
                print(f"Error predicting for time {time_str}: {e}")
                
        return predictions
        
    def predict_daily_schedule(self, date_str: Optional[str] = None, 
                             temperature: float = 22.0,
                             interval_minutes: int = 30) -> List[TimeSlotPrediction]:
        """Generate predictions for an entire day at specified intervals"""
        predictions = []
        
        # Generate time slots for the day
        start_time = datetime.strptime("00:00", "%H:%M")
        
        current_time = start_time
        while current_time.hour < 24:
            time_str = current_time.strftime("%H:%M")
            
            try:
                prediction = self.predict_for_time(time_str, date_str, temperature)
                predictions.append(prediction)
            except ValueError as e:
                print(f"Error predicting for time {time_str}: {e}")
            
            current_time += timedelta(minutes=interval_minutes)
            
        return predictions
        
    def get_device_usage_pattern(self, device_name: str, days: int = 7) -> Dict:
        """Analyze usage pattern for a specific device"""
        if device_name not in self.data_model.DEVICE_NAMES:
            raise ValueError(f"Unknown device: {device_name}")
        
        device_readings = self.data_model.get_readings_by_device(device_name)
        
        if not device_readings:
            return {"error": "No data available for this device"}
        
        # Analyze by hour
        hourly_usage = {}
        for hour in range(24):
            hour_readings = [r for r in device_readings if r.timestamp.hour == hour]
            if hour_readings:
                on_count = sum(1 for r in hour_readings if r.state == 1)
                hourly_usage[f"{hour:02d}:00"] = {
                    "usage_probability": on_count / len(hour_readings),
                    "total_readings": len(hour_readings),
                    "on_readings": on_count
                }
        
        # Overall statistics
        total_readings = len(device_readings)
        on_readings = sum(1 for r in device_readings if r.state == 1)
        
        return {
            "device_name": device_name,
            "total_readings": total_readings,
            "on_percentage": (on_readings / total_readings) * 100,
            "hourly_usage": hourly_usage,
            "peak_hours": [hour for hour, data in hourly_usage.items() 
                          if data["usage_probability"] > 0.6]
        }
        
    def optimize_energy_usage(self, time_str: str, temperature: float = 22.0,
                            priority_devices: List[str] = None) -> Dict:
        """
        Suggest energy optimization by identifying devices that can be turned off
        
        Args:
            time_str: Time for optimization analysis
            temperature: Current temperature
            priority_devices: List of devices that should not be turned off
            
        Returns:
            Dictionary with optimization suggestions
        """
        if priority_devices is None:
            priority_devices = ['Refrigerator']  # Always keep refrigerator on
        
        prediction = self.predict_for_time(time_str, temperature=temperature)
        
        # Estimate power consumption (mock values in watts)
        power_consumption = {
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
        
        current_consumption = 0
        optimized_consumption = 0
        suggestions = []
        
        for device, state in prediction.device_states.items():
            power = power_consumption.get(device, 100)
            
            if state == 1:  # Device is predicted to be ON
                current_consumption += power
                
                if device not in priority_devices:
                    confidence = prediction.confidence_scores.get(device, 0.5)
                    
                    # Suggest turning off devices with low confidence or non-essential devices
                    if confidence < 0.7 or device in ['TV', 'Fan', 'Lights']:
                        suggestions.append({
                            "device": device,
                            "action": "turn_off",
                            "potential_savings": power,
                            "confidence": confidence,
                            "reason": "Low usage confidence" if confidence < 0.7 else "Non-essential device"
                        })
                    else:
                        optimized_consumption += power
                else:
                    optimized_consumption += power
        
        total_potential_savings = sum(s["potential_savings"] for s in suggestions)
        
        return {
            "time": time_str,
            "current_predicted_consumption": current_consumption,
            "optimized_consumption": optimized_consumption,
            "potential_savings_watts": total_potential_savings,
            "potential_savings_percentage": (total_potential_savings / max(current_consumption, 1)) * 100,
            "optimization_suggestions": suggestions,
            "priority_devices": priority_devices
        }
        
    def format_prediction_output(self, prediction: TimeSlotPrediction, 
                               include_confidence: bool = False) -> str:
        """Format prediction in the requested format"""
        if include_confidence:
            formatted_output = f"{prediction.time_str}\n"
            formatted_output += f"Temperature: {prediction.temperature}°C\n"
            formatted_output += "states : {\n"
            
            for device, state in prediction.device_states.items():
                confidence = prediction.confidence_scores.get(device, 0.0)
                formatted_output += f"    {device} : {state} (confidence: {confidence:.2f}),\n"
            
            formatted_output = formatted_output.rstrip(',\n') + "\n}"
        else:
            # Simple format as requested
            formatted_output = prediction.format_prediction()
            
        return formatted_output
        
    def export_predictions_to_json(self, predictions: List[TimeSlotPrediction], 
                                 filename: str) -> None:
        """Export predictions to JSON file"""
        export_data = {
            "predictions": [pred.to_dict() for pred in predictions],
            "generated_at": datetime.now().isoformat(),
            "total_predictions": len(predictions)
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Predictions exported to {filename}")
        
    def get_model_performance(self) -> Dict:
        """Get model performance metrics"""
        if not self.is_model_loaded or len(self.data_model.data) == 0:
            return {"error": "No trained model or data available"}
        
        return self.model.evaluate_model(self.data_model)