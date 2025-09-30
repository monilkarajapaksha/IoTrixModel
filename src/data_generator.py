import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

class IoTDataGenerator:
    def __init__(self):
        self.devices = [
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
        
        # Device usage patterns (probability of being ON at different hours)
        self.device_patterns = {
            'TV': {
                'peak_hours': [19, 20, 21, 22],  # 7-10 PM
                'high_prob': 0.8,
                'low_prob': 0.2
            },
            'Washing Machine': {
                'peak_hours': [8, 9, 18, 19],  # 8-9 AM, 6-7 PM
                'high_prob': 0.6,
                'low_prob': 0.1
            },
            'AC': {
                'peak_hours': [14, 15, 16, 20, 21, 22],  # 2-4 PM, 8-10 PM
                'high_prob': 0.9,
                'low_prob': 0.3
            },
            'Refrigerator': {
                'peak_hours': list(range(24)),  # Always high probability
                'high_prob': 0.95,
                'low_prob': 0.9
            },
            'Microwave': {
                'peak_hours': [7, 8, 12, 13, 19, 20],  # Meal times
                'high_prob': 0.4,
                'low_prob': 0.05
            },
            'Water Heater': {
                'peak_hours': [6, 7, 8, 19, 20, 21],  # Morning and evening
                'high_prob': 0.7,
                'low_prob': 0.2
            },
            'Lights': {
                'peak_hours': [18, 19, 20, 21, 22],  # Evening
                'high_prob': 0.9,
                'low_prob': 0.1
            },
            'Fan': {
                'peak_hours': [14, 15, 16, 21, 22, 23],  # Hot hours and night
                'high_prob': 0.8,
                'low_prob': 0.3
            },
            'Dishwasher': {
                'peak_hours': [20, 21, 22],  # After dinner
                'high_prob': 0.5,
                'low_prob': 0.1
            },
            'Electric Stove': {
                'peak_hours': [7, 8, 12, 13, 19, 20],  # Cooking times
                'high_prob': 0.6,
                'low_prob': 0.05
            }
        }
    
    def generate_temperature(self, hour, base_temp=22):
        """Generate realistic temperature based on time of day"""
        # Temperature variation throughout the day
        temp_variation = 5 * np.sin((hour - 6) * np.pi / 12)
        noise = np.random.normal(0, 1)
        return round(base_temp + temp_variation + noise, 1)
    
    def get_device_state_probability(self, device, hour, temperature):
        """Calculate probability of device being ON based on time and temperature"""
        pattern = self.device_patterns[device]
        
        if hour in pattern['peak_hours']:
            base_prob = pattern['high_prob']
        else:
            base_prob = pattern['low_prob']
        
        # Temperature adjustments
        if device == 'AC' and temperature > 25:
            base_prob += 0.2
        elif device == 'AC' and temperature < 18:
            base_prob -= 0.3
        elif device == 'Fan' and temperature > 26:
            base_prob += 0.3
        elif device == 'Water Heater' and temperature < 15:
            base_prob += 0.2
        
        return max(0, min(1, base_prob))
    
    def generate_dataset(self, days=30, start_date=None):
        """Generate IoT dataset for specified number of days"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        data = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            for hour in range(24):
                for minute in [0, 30]:  # Every 30 minutes
                    timestamp = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    temperature = self.generate_temperature(hour)
                    
                    for device in self.devices:
                        prob = self.get_device_state_probability(device, hour, temperature)
                        state = 1 if random.random() < prob else 0
                        
                        data.append({
                            'timestamp': timestamp,
                            'date': timestamp.strftime('%Y-%m-%d'),
                            'time': timestamp.strftime('%H:%M'),
                            'day_of_week': timestamp.strftime('%A'),
                            'month': timestamp.strftime('%B'),
                            'device_name': device,
                            'state': state,
                            'temperature': temperature
                        })
        
        return pd.DataFrame(data)
    
    def save_dataset(self, df, filename='iot_data.csv'):
        """Save dataset to CSV file"""
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return filepath
    
    def generate_and_save(self, days=30, filename='iot_data.csv'):
        """Generate and save dataset"""
        print(f"Generating IoT dataset for {days} days...")
        df = self.generate_dataset(days)
        filepath = self.save_dataset(df, filename)
        
        print(f"\nDataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Devices: {len(self.devices)}")
        print(f"Average temperature: {df['temperature'].mean():.1f}°C")
        
        return df, filepath

if __name__ == "__main__":
    generator = IoTDataGenerator()
    df, filepath = generator.generate_and_save(days=30)
    
    # Show sample data
    print("\nSample data:")
    print(df.head(10))