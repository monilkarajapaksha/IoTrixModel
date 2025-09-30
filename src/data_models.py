from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class DeviceReading:
    """Represents a single device reading at a specific time"""
    timestamp: datetime
    device_name: str
    state: int  # 0 = OFF, 1 = ON
    temperature: float
    date: str
    time: str
    day_of_week: str
    month: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'device_name': self.device_name,
            'state': self.state,
            'temperature': self.temperature,
            'date': self.date,
            'time': self.time,
            'day_of_week': self.day_of_week,
            'month': self.month
        }

@dataclass
class DeviceState:
    """Represents the state of a single device"""
    device_name: str
    state: int  # 0 = OFF, 1 = ON
    confidence: float = 0.0  # Prediction confidence (0-1)
    
    def __str__(self) -> str:
        return f"{self.device_name}: {self.state}"

@dataclass
class TimeSlotPrediction:
    """Represents prediction for all devices at a specific time"""
    timestamp: datetime
    time_str: str
    device_states: Dict[str, int]
    temperature: float
    confidence_scores: Dict[str, float]
    
    def format_prediction(self) -> str:
        """Format prediction in the requested format"""
        formatted_states = {device: state for device, state in self.device_states.items()}
        return f"{self.time_str}\nstates : {formatted_states}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'time': self.time_str,
            'states': self.device_states,
            'temperature': self.temperature,
            'confidence_scores': self.confidence_scores
        }

@dataclass
class ModelFeatures:
    """Features used for ML model training/prediction"""
    hour: int
    minute: int
    day_of_week: int  # 0 = Monday, 6 = Sunday
    month: int
    temperature: float
    is_weekend: bool
    is_peak_hour: bool  # Based on general peak electricity usage
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for model input"""
        return {
            'hour': self.hour,
            'minute': self.minute,
            'day_of_week': self.day_of_week,
            'month': self.month,
            'temperature': self.temperature,
            'is_weekend': int(self.is_weekend),
            'is_peak_hour': int(self.is_peak_hour)
        }

class IoTDataModel:
    """Main data model class for handling IoT data operations"""
    
    DEVICE_NAMES = [
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
    
    PEAK_HOURS = [7, 8, 18, 19, 20, 21, 22]  # Typical peak electricity hours
    
    def __init__(self):
        self.data: List[DeviceReading] = []
        
    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Load data from pandas DataFrame"""
        self.data = []
        for _, row in df.iterrows():
            reading = DeviceReading(
                timestamp=pd.to_datetime(row['timestamp']),
                device_name=row['device_name'],
                state=int(row['state']),
                temperature=float(row['temperature']),
                date=row['date'],
                time=row['time'],
                day_of_week=row['day_of_week'],
                month=row['month']
            )
            self.data.append(reading)
    
    def get_readings_by_device(self, device_name: str) -> List[DeviceReading]:
        """Get all readings for a specific device"""
        return [reading for reading in self.data if reading.device_name == device_name]
    
    def get_readings_by_time_range(self, start_time: datetime, end_time: datetime) -> List[DeviceReading]:
        """Get readings within a time range"""
        return [reading for reading in self.data 
                if start_time <= reading.timestamp <= end_time]
    
    def extract_features_from_timestamp(self, timestamp: datetime, temperature: float) -> ModelFeatures:
        """Extract ML features from timestamp and temperature"""
        return ModelFeatures(
            hour=timestamp.hour,
            minute=timestamp.minute,
            day_of_week=timestamp.weekday(),
            month=timestamp.month,
            temperature=temperature,
            is_weekend=timestamp.weekday() >= 5,
            is_peak_hour=timestamp.hour in self.PEAK_HOURS
        )
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare data for ML model training"""
        training_data = []
        
        for reading in self.data:
            features = self.extract_features_from_timestamp(reading.timestamp, reading.temperature)
            
            # Create a row with features and target (device state)
            row = features.to_dict()
            row['device_name'] = reading.device_name
            row['target'] = reading.state
            
            training_data.append(row)
        
        return pd.DataFrame(training_data)
    
    def get_device_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each device"""
        stats = {}
        
        for device in self.DEVICE_NAMES:
            device_readings = self.get_readings_by_device(device)
            if device_readings:
                total_readings = len(device_readings)
                on_readings = sum(1 for r in device_readings if r.state == 1)
                
                stats[device] = {
                    'total_readings': total_readings,
                    'on_readings': on_readings,
                    'off_readings': total_readings - on_readings,
                    'on_percentage': (on_readings / total_readings) * 100,
                    'avg_temperature_when_on': sum(r.temperature for r in device_readings if r.state == 1) / max(on_readings, 1)
                }
        
        return stats

# Utility functions
def create_time_slot_prediction(time_str: str, device_states: Dict[str, int], 
                               temperature: float = 22.0, 
                               confidence_scores: Dict[str, float] = None) -> TimeSlotPrediction:
    """Helper function to create TimeSlotPrediction"""
    if confidence_scores is None:
        confidence_scores = {device: 0.8 for device in device_states.keys()}
    
    # Parse time string to create timestamp (assuming today's date)
    from datetime import datetime, time
    hour, minute = map(int, time_str.split(':'))
    timestamp = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    return TimeSlotPrediction(
        timestamp=timestamp,
        time_str=time_str,
        device_states=device_states,
        temperature=temperature,
        confidence_scores=confidence_scores
    )