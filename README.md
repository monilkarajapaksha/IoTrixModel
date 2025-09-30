# IoTrix Model - Autonomous Power Agent

A smart IoT device state prediction system that uses machine learning to predict whether home appliances should be on or off based on time patterns, temperature, and historical usage data.

## 🏠 Project Overview

This system can predict the optimal states for 10 different house appliances:

- TV
- Washing Machine
- AC (Air Conditioner)
- Refrigerator
- Microwave
- Water Heater
- Lights
- Fan
- Dishwasher
- Electric Stove

## 🚀 Features

- **Smart Predictions**: ML-based predictions considering time, temperature, and usage patterns
- **Energy Optimization**: Recommendations for reducing power consumption
- **Real-time Analysis**: Predict device states for any time of day
- **Pattern Recognition**: Analyze usage patterns for different devices
- **Interactive Mode**: Test predictions for custom times
- **Structured Data**: Well-organized project with modular components

## 📁 Project Structure

```
IoTrixModel/
├── src/
│   ├── data_generator.py      # Generate synthetic IoT data
│   ├── data_models.py         # Data structures and models
│   ├── prediction_model.py    # ML prediction model
│   └── prediction_service.py  # Prediction service layer
├── data/                      # Generated datasets
├── models/                    # Trained ML models
├── config/
│   └── config.py             # Configuration settings
├── utils/
│   └── helpers.py            # Utility functions
├── output/                   # Output files and reports
├── main.py                   # Main application
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🛠️ Installation

1. **Clone or download the project**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Run the Main Demo

```bash
python main.py
```

This will:

1. Generate a synthetic IoT dataset (30 days of data)
2. Train the ML prediction model
3. Show prediction for 7:30 PM (as requested)
4. Demonstrate additional features

### Expected Output Format

The system outputs predictions in the requested format:

```
7:30 p.m.
states : {
    TV : 1,
    Washing Machine : 0,
    AC: 1,
    Refrigerator: 1,
    Microwave: 0,
    Water Heater: 1,
    Lights: 1,
    Fan: 0,
    Dishwasher: 1,
    Electric Stove: 0
}
```

Where:

- `1` = Device should be ON
- `0` = Device should be OFF

### Interactive Mode

After running the main demo, you can use interactive mode to test predictions for any time:

```
Enter time: 19:30
Enter temperature (°C, default 22): 24

7:30 p.m.
states : {
    TV : 1,
    Washing Machine : 0,
    AC: 1
    ...
}
```

## 🧠 How It Works

### 1. Data Generation

- Creates realistic IoT data with time-based patterns
- Considers device-specific usage patterns (e.g., TV usage peaks in evening)
- Incorporates temperature effects (e.g., AC usage increases with temperature)

### 2. Machine Learning Model

- Uses Random Forest classifier for each device
- Features include: hour, minute, day of week, temperature, weekend indicator
- Trained on historical usage patterns with 80/20 train/test split

### 3. Prediction Logic

- Time-based patterns (peak hours for different devices)
- Temperature influence (AC, fans, water heater)
- Day type consideration (weekday vs weekend)
- Confidence scoring for each prediction

### 4. Energy Optimization

- Identifies devices that can be turned off to save energy
- Considers device priority (e.g., refrigerator always stays on)
- Calculates potential power savings

## 📊 Sample Data Structure

The system works with data in this format:

| timestamp           | device_name | state | temperature | date       | time  | day_of_week | month   |
| ------------------- | ----------- | ----- | ----------- | ---------- | ----- | ----------- | ------- |
| 2024-01-01 19:30:00 | TV          | 1     | 24.0        | 2024-01-01 | 19:30 | Monday      | January |
| 2024-01-01 19:30:00 | AC          | 1     | 24.0        | 2024-01-01 | 19:30 | Monday      | January |

## 🎛️ Configuration

Key settings can be modified in `config/config.py`:

- Device power consumption values
- Temperature thresholds
- Peak usage hours
- Model parameters
- Priority devices (that shouldn't be auto-turned off)

## 🔧 Customization

### Adding New Devices

1. Add device name to `DEVICES` list in `config/config.py`
2. Add power consumption value in `DEVICE_POWER_CONSUMPTION`
3. Optionally add usage patterns in `data_generator.py`

### Modifying Prediction Logic

- Edit device patterns in `IoTDataGenerator.device_patterns`
- Adjust temperature thresholds in configuration
- Modify rule-based fallback logic in `prediction_model.py`

## 📈 Model Performance

The system achieves good accuracy by:

- Learning from realistic usage patterns
- Considering multiple factors (time, temperature, day type)
- Using ensemble methods (Random Forest)
- Providing confidence scores for predictions

## 🔮 Future Enhancements

- Real-time data integration from actual IoT devices
- Mobile app interface
- Smart home automation integration
- Advanced energy optimization algorithms
- Weather data integration
- User preference learning

## 🤝 Contributing

This project is designed for educational and demonstration purposes. Feel free to:

- Add new devices or features
- Improve prediction algorithms
- Enhance the user interface
- Add real IoT device integrations

## 📝 License

This project is for educational use. Feel free to modify and adapt for your needs.

---

**IoTrix Model** - Smart IoT Device State Prediction System
_Predicting the future of home automation_ 🏠🤖
