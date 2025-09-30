import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config.config import LOGGING_CONFIG, PATHS

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        datefmt=LOGGING_CONFIG['date_format']
    )
    return logging.getLogger(__name__)

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        PATHS['data_dir'],
        PATHS['models_dir'],
        PATHS['output_dir']
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def validate_time_format(time_str: str) -> bool:
    """Validate time string format (HH:MM)"""
    try:
        hour, minute = map(int, time_str.split(':'))
        return 0 <= hour <= 23 and 0 <= minute <= 59
    except (ValueError, AttributeError):
        return False

def validate_date_format(date_str: str) -> bool:
    """Validate date string format (YYYY-MM-DD)"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def parse_time_string(time_str: str) -> tuple:
    """Parse time string and return (hour, minute)"""
    if not validate_time_format(time_str):
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM format.")
    
    hour, minute = map(int, time_str.split(':'))
    return hour, minute

def convert_to_12_hour_format(time_str: str) -> str:
    """Convert 24-hour time to 12-hour format with AM/PM"""
    hour, minute = parse_time_string(time_str)
    
    if hour == 0:
        return f"12:{minute:02d} AM"
    elif hour < 12:
        return f"{hour}:{minute:02d} AM"
    elif hour == 12:
        return f"12:{minute:02d} PM"
    else:
        return f"{hour-12}:{minute:02d} PM"

def get_time_category(hour: int) -> str:
    """Categorize time of day"""
    if 6 <= hour <= 10:
        return "Morning"
    elif 11 <= hour <= 17:
        return "Afternoon"
    elif 18 <= hour <= 22:
        return "Evening"
    else:
        return "Night"

def calculate_energy_cost(power_watts: float, hours: float, rate_per_kwh: float = 0.12) -> float:
    """Calculate energy cost based on power consumption"""
    kwh = (power_watts * hours) / 1000
    return kwh * rate_per_kwh

def format_power_consumption(watts: float) -> str:
    """Format power consumption with appropriate units"""
    if watts >= 1000:
        return f"{watts/1000:.1f} kW"
    else:
        return f"{watts:.0f} W"

def generate_time_slots(start_hour: int = 0, end_hour: int = 23, 
                       interval_minutes: int = 30) -> List[str]:
    """Generate list of time slots for a day"""
    time_slots = []
    
    current_time = datetime.strptime("00:00", "%H:%M")
    current_time = current_time.replace(hour=start_hour)
    end_time = datetime.strptime("23:59", "%H:%M")
    end_time = end_time.replace(hour=end_hour)
    
    while current_time <= end_time:
        time_slots.append(current_time.strftime("%H:%M"))
        current_time += timedelta(minutes=interval_minutes)
    
    return time_slots

def save_dict_to_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"Data saved to {filepath}")

def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)

def create_summary_table(data: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table from dictionary data"""
    if not data:
        return pd.DataFrame()
    
    # Handle nested dictionaries
    flattened_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flattened_data[f"{key}_{sub_key}"] = sub_value
        else:
            flattened_data[key] = value
    
    return pd.DataFrame([flattened_data])

def format_prediction_summary(predictions: List[Dict]) -> str:
    """Format prediction results for display"""
    if not predictions:
        return "No predictions available"
    
    summary = "IoT Device Prediction Summary\n"
    summary += "=" * 40 + "\n\n"
    
    for pred in predictions:
        time_str = pred.get('time', 'Unknown')
        states = pred.get('states', {})
        temperature = pred.get('temperature', 0)
        
        summary += f"Time: {time_str}\n"
        summary += f"Temperature: {temperature}°C\n"
        summary += "Device States:\n"
        
        for device, state in states.items():
            status = "ON" if state == 1 else "OFF"
            summary += f"  {device}: {status}\n"
        
        summary += "\n" + "-" * 30 + "\n\n"
    
    return summary

def get_file_size(filepath: str) -> str:
    """Get human-readable file size"""
    if not os.path.exists(filepath):
        return "File not found"
    
    size_bytes = os.path.getsize(filepath)
    
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def check_system_requirements() -> Dict[str, bool]:
    """Check if system meets requirements"""
    requirements = {
        'python_version': sys.version_info >= (3, 7),
        'pandas_available': False,
        'sklearn_available': False,
        'numpy_available': False,
        'joblib_available': False
    }
    
    try:
        import pandas
        requirements['pandas_available'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        requirements['sklearn_available'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        requirements['numpy_available'] = True
    except ImportError:
        pass
    
    try:
        import joblib
        requirements['joblib_available'] = True
    except ImportError:
        pass
    
    return requirements

def print_system_info():
    """Print system information and requirements"""
    print("IoTrix Model System Information")
    print("=" * 40)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current Working Directory: {os.getcwd()}")
    
    requirements = check_system_requirements()
    print("\nDependency Check:")
    for requirement, status in requirements.items():
        status_str = "✓" if status else "✗"
        print(f"  {requirement}: {status_str}")
    
    print()

def create_project_structure():
    """Create the complete project structure if it doesn't exist"""
    structure = {
        'data': [],
        'models': [],
        'src': ['__init__.py'],
        'config': ['__init__.py'],
        'utils': ['__init__.py'],
        'output': [],
        'tests': ['__init__.py'],
        'docs': []
    }
    
    for directory, files in structure.items():
        # Create directory
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Create files
        for file in files:
            filepath = os.path.join(directory, file)
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    if file == '__init__.py':
                        f.write(f'"""IoTrix Model - {directory} module"""\n')
                print(f"Created file: {filepath}")

if __name__ == "__main__":
    print_system_info()
    ensure_directories()
    create_project_structure()