"""
Simple IoT Device State Predictor
Just enter time and get predictions!
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'config'))

from src.data_generator import IoTDataGenerator
from src.prediction_service import IoTPredictionService
from config.config import PATHS

def setup_system():
    """Quick setup - create data and model if needed"""
    service = IoTPredictionService()
    
    # File paths
    model_path = os.path.join(PATHS['models_dir'], PATHS['default_model_file'])
    data_path = os.path.join(PATHS['data_dir'], PATHS['default_data_file'])
    
    # Create directories if needed
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    os.makedirs(PATHS['models_dir'], exist_ok=True)
    
    # Generate data if doesn't exist
    if not os.path.exists(data_path):
        print("Creating dataset...")
        generator = IoTDataGenerator()
        generator.generate_and_save(days=30, filename=PATHS['default_data_file'])
    
    # Train model if doesn't exist
    if not os.path.exists(model_path):
        print("Training model...")
        service.load_data(data_path)
        service.train_model()
        service.save_model(model_path)
    else:
        service.load_data(data_path)
        service.load_trained_model(model_path)
    
    return service

def main():
    print("IoTrix - Smart Home Device Predictor")
    print("="*50)
    
    # Setup system
    service = setup_system()
    print("✅ System ready!")
    print()
    
    print("Enter time to predict device states")
    print("Format: HH:MM (24-hour)")
    print("Examples: 07:30, 19:30, 22:00")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            time_input = input("Time: ").strip()
            
            if time_input.lower() in ['quit', 'exit', 'q', '']:
                break
            
            # Make prediction
            prediction = service.predict_for_time(time_input, temperature=22.0)
            
            # Show result in requested format
            print("\n" + prediction.format_prediction())
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please use HH:MM format (e.g., 19:30)")
            print()
    
    print("Goodbye! 👋")

if __name__ == "__main__":
    main()