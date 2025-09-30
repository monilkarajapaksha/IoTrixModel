"""
IoTrix Model - Autonomous Power Agent
Smart IoT Device State Prediction System

Enter time to get device state predictions
"""

import os
import sys
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'config'))
sys.path.append(os.path.join(project_root, 'utils'))

from src.data_generator import IoTDataGenerator
from src.prediction_service import IoTPredictionService
from utils.helpers import setup_logging, ensure_directories
from config.config import PATHS, DEVICES

def initialize_system():
    """Initialize the prediction system"""
    print("🔧 Initializing IoTrix Model...")
    
    # Setup directories
    ensure_directories()
    
    # Initialize service
    service = IoTPredictionService()
    
    # Check if model and data exist
    model_path = os.path.join(PATHS['models_dir'], PATHS['default_model_file'])
    data_path = os.path.join(PATHS['data_dir'], PATHS['default_data_file'])
    
    if not os.path.exists(data_path):
        print("📊 Generating IoT dataset...")
        generator = IoTDataGenerator()
        df, data_path = generator.generate_and_save(days=30, filename=PATHS['default_data_file'])
        print(f"✅ Dataset created with {len(df):,} records")
    
    if not os.path.exists(model_path):
        print("🤖 Training prediction model...")
        service.load_data(data_path)
        accuracies = service.train_model()
        service.save_model(model_path)
        avg_accuracy = sum(accuracies.values()) / len(accuracies)
        print(f"✅ Model trained with {avg_accuracy:.1%} average accuracy")
    else:
        print("📂 Loading existing model...")
        service.load_data(data_path)
        service.load_trained_model(model_path)
        print("✅ Model loaded successfully")
    
    return service

def get_prediction(service, time_input, temp_input=None):
    """Get prediction for given time"""
    try:
        # Default temperature if not provided
        temperature = float(temp_input) if temp_input else 22.0
        
        # Make prediction
        prediction = service.predict_for_time(time_input, temperature=temperature)
        
        # Format and display result
        print("\n" + "="*50)
        print(prediction.format_prediction())
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main application function"""
    print("=" * 60)
    print("          IoTrix Model - Autonomous Power Agent")
    print("       Smart IoT Device State Prediction System")
    print("=" * 60)
    print()
    
    try:
        # Initialize system
        service = initialize_system()
        print()
        
        print("🎯 IoT Device State Predictor")
        print("-" * 40)
        print("Enter time in HH:MM format (24-hour)")
        print("Examples: 07:30, 13:45, 19:30, 22:00")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                # Get time input
                time_input = input("Enter time: ").strip()
                
                if time_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not time_input:
                    continue
                
                # Get temperature (optional)
                temp_input = input("Enter temperature (°C, press Enter for default 22°C): ").strip()
                
                # Get and display prediction
                get_prediction(service, time_input, temp_input)
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                print()
        
        print("Thank you for using IoTrix Model! 🚀")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()