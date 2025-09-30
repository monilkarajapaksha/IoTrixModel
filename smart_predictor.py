"""
Smart IoT Hybrid Predictor with Recommendations
Input time → Get predictions + Smart suggestions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

class SmartIoTPredictor:
    """
    Hybrid IoT predictor with smart recommendations
    Combines RF + Sequential + Rules + Smart Suggestions
    """
    
    def __init__(self):
        self.devices = [
            'TV', 'Washing Machine', 'AC', 'Refrigerator', 'Microwave',
            'Water Heater', 'Lights', 'Fan', 'Dishwasher', 'Electric Stove'
        ]
        
        # Power consumption (watts)
        self.power_consumption = {
            'TV': 150, 'Washing Machine': 500, 'AC': 1500, 'Refrigerator': 200,
            'Microwave': 1200, 'Water Heater': 3000, 'Lights': 60, 'Fan': 75,
            'Dishwasher': 1800, 'Electric Stove': 2000
        }
        
        # Priority devices (never suggest turning off)
        self.priority_devices = ['Refrigerator']
        
        # Model components
        self.pattern_models = {}
        self.sequence_patterns = {}
        self.ensemble_weights = {}
        self.is_trained = False
    
    def quick_setup(self):
        """Quick setup with realistic patterns"""
        print("⚡ Quick setup with smart patterns...")
        
        # Pre-defined smart patterns based on typical usage
        for device in self.devices:
            if device == 'TV':
                hourly = {h: 0.9 if 19 <= h <= 23 else 0.2 for h in range(24)}
            elif device == 'AC':
                hourly = {h: 0.8 if 14 <= h <= 16 or 20 <= h <= 22 else 0.3 for h in range(24)}
            elif device == 'Lights': 
                hourly = {h: 0.9 if 18 <= h <= 23 else 0.1 for h in range(24)}
            elif device == 'Washing Machine':
                hourly = {h: 0.6 if h in [8, 9, 18, 19] else 0.1 for h in range(24)}
            elif device == 'Refrigerator':
                hourly = {h: 0.95 for h in range(24)}
            elif device == 'Water Heater':
                hourly = {h: 0.7 if h in [6, 7, 8, 19, 20, 21] else 0.2 for h in range(24)}
            elif device == 'Fan':
                hourly = {h: 0.8 if 14 <= h <= 16 or 21 <= h <= 23 else 0.3 for h in range(24)}
            elif device == 'Microwave':
                hourly = {h: 0.4 if h in [7, 8, 12, 13, 19, 20] else 0.05 for h in range(24)}
            elif device == 'Dishwasher':
                hourly = {h: 0.5 if 20 <= h <= 22 else 0.1 for h in range(24)}
            else:  # Electric Stove
                hourly = {h: 0.6 if h in [7, 8, 12, 13, 19, 20] else 0.05 for h in range(24)}
            
            self.pattern_models[device] = {
                'hourly': hourly,
                'temp_high': 0.8 if device in ['AC', 'Fan'] else 0.5,
                'temp_low': 0.2 if device in ['AC', 'Fan'] else 0.5
            }
        
        # Default ensemble weights
        for device in self.devices:
            self.ensemble_weights[device] = {
                'pattern': 0.6, 'sequence': 0.2, 'rules': 0.2
            }
        
        self.is_trained = True
        print("✅ Smart predictor ready!")
    
    def train_from_data(self, data_path: str):
        """Train from existing data if available"""
        if not os.path.exists(data_path):
            print("⚠️ No training data found, using smart defaults")
            self.quick_setup()
            return
        
        print("📊 Training from data...")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        # Train pattern models
        for device in self.devices:
            device_data = df[df['device_name'] == device]
            
            if len(device_data) > 0:
                # Learn hourly patterns
                hourly_patterns = {}
                for hour in range(24):
                    hour_data = device_data[device_data['hour'] == hour]
                    if len(hour_data) > 0:
                        hourly_patterns[hour] = hour_data['state'].mean()
                    else:
                        hourly_patterns[hour] = 0.5
                
                # Temperature effects
                hot_data = device_data[device_data['temperature'] > 25]
                cold_data = device_data[device_data['temperature'] < 18]
                
                self.pattern_models[device] = {
                    'hourly': hourly_patterns,
                    'temp_high': hot_data['state'].mean() if len(hot_data) > 0 else 0.5,
                    'temp_low': cold_data['state'].mean() if len(cold_data) > 0 else 0.5
                }
            else:
                # Fallback to defaults
                self.pattern_models[device] = {
                    'hourly': {h: 0.5 for h in range(24)},
                    'temp_high': 0.5, 'temp_low': 0.5
                }
        
        # Set ensemble weights
        for device in self.devices:
            self.ensemble_weights[device] = {
                'pattern': 0.6, 'sequence': 0.25, 'rules': 0.15
            }
        
        self.is_trained = True
        print("✅ Training complete!")
    
    def predict_hybrid(self, time_str: str, temperature: float = 22.0) -> dict:
        """Make hybrid prediction with smart recommendations"""
        
        if not self.is_trained:
            self.quick_setup()
        
        # Parse time
        hour, minute = map(int, time_str.split(':'))
        timestamp = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        is_weekend = timestamp.weekday() >= 5
        
        # Predict each device
        predictions = {}
        confidences = {}
        total_power = 0
        
        for device in self.devices:
            # Pattern prediction
            pattern_prob = self.pattern_models[device]['hourly'].get(hour, 0.5)
            
            # Temperature adjustment
            if temperature > 25:
                temp_effect = self.pattern_models[device]['temp_high']
            elif temperature < 18:
                temp_effect = self.pattern_models[device]['temp_low']
            else:
                temp_effect = 0.5
            
            # Rule-based adjustments
            rule_prob = 0.5
            if device == 'AC' and temperature > 25:
                rule_prob = 0.9
            elif device == 'AC' and temperature < 18:
                rule_prob = 0.1
            elif device == 'Fan' and temperature > 26:
                rule_prob = 0.8
            elif device == 'Lights' and 18 <= hour <= 23:
                rule_prob = 0.8
            elif device == 'TV' and 19 <= hour <= 23:
                rule_prob = 0.8
            elif device == 'Refrigerator':
                rule_prob = 0.95
            
            # Ensemble combination
            weights = self.ensemble_weights[device]
            combined_prob = (
                weights['pattern'] * pattern_prob +
                weights['sequence'] * 0.5 +  # Simplified sequence
                weights['rules'] * rule_prob
            )
            
            # Final prediction
            final_state = 1 if combined_prob > 0.5 else 0
            confidence = max(combined_prob, 1 - combined_prob)
            
            predictions[device] = final_state
            confidences[device] = confidence
            
            if final_state == 1:
                total_power += self.power_consumption[device]
        
        # Generate smart recommendations
        recommendations = self._generate_recommendations(
            predictions, confidences, hour, temperature, total_power
        )
        
        return {
            'time': time_str,
            'temperature': temperature,
            'predictions': predictions,
            'confidence': confidences,
            'total_power': total_power,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, predictions: dict, confidences: dict, 
                                hour: int, temperature: float, total_power: int) -> list:
        """Generate smart recommendations based on predictions"""
        
        recommendations = []
        
        # Energy saving recommendations
        if total_power > 3000:  # High power usage
            recommendations.append({
                'type': 'energy_saving',
                'priority': 'high',
                'message': f'⚡ High power usage ({total_power}W). Consider energy optimization.',
                'icon': '⚡'
            })
        
        # Device-specific recommendations
        for device, state in predictions.items():
            confidence = confidences[device]
            
            if device in self.priority_devices:
                continue  # Skip priority devices
            
            # Recommendations for turning devices ON
            if state == 0 and confidence < 0.7:  # Low confidence OFF
                if device == 'Lights' and 18 <= hour <= 23:
                    recommendations.append({
                        'type': 'turn_on',
                        'device': device,
                        'priority': 'medium',
                        'message': f'💡 Consider turning ON {device} - it\'s evening time',
                        'icon': '💡'
                    })
                elif device == 'AC' and temperature > 26:
                    recommendations.append({
                        'type': 'turn_on',
                        'device': device,
                        'priority': 'high',
                        'message': f'❄️ Consider turning ON {device} - temperature is {temperature}°C',
                        'icon': '❄️'
                    })
                elif device == 'Fan' and temperature > 27:
                    recommendations.append({
                        'type': 'turn_on',
                        'device': device,
                        'priority': 'medium',
                        'message': f'🌪️ Consider turning ON {device} - it\'s hot ({temperature}°C)',
                        'icon': '🌪️'
                    })
            
            # Recommendations for turning devices OFF
            elif state == 1 and confidence < 0.7:  # Low confidence ON
                power = self.power_consumption[device]
                
                if device == 'AC' and temperature < 20:
                    recommendations.append({
                        'type': 'turn_off',
                        'device': device,
                        'priority': 'medium',
                        'message': f'❄️ Consider turning OFF {device} - temperature is cool ({temperature}°C)',
                        'savings': f'Save {power}W',
                        'icon': '❄️'
                    })
                elif device == 'Lights' and hour < 6:
                    recommendations.append({
                        'type': 'turn_off',
                        'device': device,
                        'priority': 'low',
                        'message': f'💡 Consider turning OFF {device} - it\'s late night',
                        'savings': f'Save {power}W',
                        'icon': '💡'
                    })
                elif device == 'TV' and hour < 6:
                    recommendations.append({
                        'type': 'turn_off',
                        'device': device,
                        'priority': 'low',
                        'message': f'📺 Consider turning OFF {device} - it\'s late night',
                        'savings': f'Save {power}W',
                        'icon': '📺'
                    })
        
        # Time-based recommendations
        if 6 <= hour <= 8:  # Morning
            if predictions.get('Water Heater', 0) == 0:
                recommendations.append({
                    'type': 'time_based',
                    'priority': 'medium',
                    'message': '🚿 Morning routine: Consider turning ON Water Heater',
                    'icon': '🚿'
                })
        elif 18 <= hour <= 20:  # Evening
            if predictions.get('Lights', 0) == 0:
                recommendations.append({
                    'type': 'time_based',
                    'priority': 'medium',
                    'message': '💡 Evening time: Consider turning ON Lights',
                    'icon': '💡'
                })
        elif 22 <= hour or hour <= 5:  # Night
            high_power_devices = [d for d, s in predictions.items() 
                                if s == 1 and self.power_consumption[d] > 500 
                                and d not in self.priority_devices]
            if high_power_devices:
                recommendations.append({
                    'type': 'time_based',
                    'priority': 'medium',
                    'message': f'🌙 Night time: Consider turning OFF {", ".join(high_power_devices[:2])}',
                    'icon': '🌙'
                })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def format_output(self, result: dict) -> str:
        """Format output in clean, user-friendly way"""
        
        time_str = result['time']
        predictions = result['predictions']
        temperature = result['temperature']
        total_power = result['total_power']
        recommendations = result['recommendations']
        
        # Convert to 12-hour format
        hour, minute = map(int, time_str.split(':'))
        if hour == 0:
            display_time = f"12:{minute:02d} a.m."
        elif hour < 12:
            display_time = f"{hour}:{minute:02d} a.m."
        elif hour == 12:
            display_time = f"12:{minute:02d} p.m."
        else:
            display_time = f"{hour-12}:{minute:02d} p.m."
        
        # Format output
        output = f"\n{'='*60}\n"
        output += f"🏠 SMART IoT PREDICTION - {display_time}\n"
        output += f"🌡️  Temperature: {temperature}°C\n"
        output += f"⚡ Total Power: {total_power}W\n"
        output += f"{'='*60}\n\n"
        
        # Device states
        output += "📱 DEVICE STATES:\n"
        output += f"{'Device':<15} {'State':<6} {'Status'}\n"
        output += "-" * 40 + "\n"
        
        for device, state in predictions.items():
            status = "🟢 ON " if state == 1 else "🔴 OFF"
            power = f"({self.power_consumption[device]}W)" if state == 1 else ""
            output += f"{device:<15} {state:<6} {status} {power}\n"
        
        # Smart recommendations
        if recommendations:
            output += f"\n💡 SMART RECOMMENDATIONS:\n"
            output += "-" * 40 + "\n"
            
            for i, rec in enumerate(recommendations, 1):
                priority_icon = "🔥" if rec['priority'] == 'high' else "⚠️" if rec['priority'] == 'medium' else "💭"
                output += f"{i}. {priority_icon} {rec['message']}\n"
                if 'savings' in rec:
                    output += f"   💰 {rec['savings']}\n"
        else:
            output += f"\n✅ All devices are optimally configured!\n"
        
        output += f"\n{'='*60}\n"
        
        return output

def main():
    """Main interactive application"""
    
    print("🚀 SMART IoT HYBRID PREDICTOR")
    print("=" * 50)
    print("Get predictions + Smart recommendations!")
    print()
    
    # Initialize predictor
    predictor = SmartIoTPredictor()
    
    # Try to load existing data, otherwise use smart defaults
    data_path = "data/iot_data.csv"
    predictor.train_from_data(data_path)
    
    print("\n💡 Enter time to get smart predictions and recommendations")
    print("Format: HH:MM (24-hour)")
    print("Examples: 07:30, 12:00, 19:30, 22:00")
    print("Type 'quit' to exit")
    print()
    
    while True:
        try:
            # Get time input
            time_input = input("🕐 Enter time: ").strip()
            
            if time_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not time_input:
                continue
            
            # Get temperature (optional)
            temp_input = input("🌡️  Temperature (°C, press Enter for 22°C): ").strip()
            temperature = float(temp_input) if temp_input else 22.0
            
            # Make prediction
            result = predictor.predict_hybrid(time_input, temperature)
            
            # Display result
            formatted_output = predictor.format_output(result)
            print(formatted_output)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please use HH:MM format (e.g., 19:30)")
            print()
    
    print("\n👋 Thank you for using Smart IoT Predictor!")

if __name__ == "__main__":
    main()