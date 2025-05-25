import os
import sys
import pandas as pd
import random  # Add this import for the fallback forecast
from datetime import datetime, timedelta

# Import the prediction systems
from models.short_term_prediction import ShortTermPredictionSystem

class WeatherPredictionSystem:
    """Main weather prediction system that coordinates all prediction types"""
    
    def __init__(self):
        """Initialize the weather prediction system"""
        self.cities = {
            "Bangalore": {
                "coords": (12.9716, 77.5946),
                "country": "India",
                "timezone": "Asia/Kolkata"
            }
        }
        
        # Create output directory
        self.output_dir = "d:\\lastone\\weather\\output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize models directory
        self.models_dir = "d:\\lastone\\weather\\models\\trained"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Placeholder for models (will be created during training)
        self.bayesian_nn = None
        self.physics_nn = None
        self.temporal_transformer = None
        self.spatial_transformer = None
        
        print("Weather Prediction System initialized")

# Add this function to prepare prediction input
# Fix the prepare_prediction_input function to accept self as the first parameter
def prepare_prediction_input(self, historical_data):
    """Prepare input data for prediction from historical data"""
    # Simple feature extraction from historical data
    # This is a basic implementation - you can enhance this with more features
    features = historical_data[['tmax', 'tmin', 'tavg', 'prcp', 'wspd', 'pres', 'humidity']].copy()
    
    # Add some derived features
    features['temp_range'] = features['tmax'] - features['tmin']
    
    # Add rolling averages (7-day window)
    for col in ['tavg', 'prcp', 'wspd', 'humidity']:
        features[f'{col}_7day_avg'] = features[col].rolling(window=7, min_periods=1).mean()
    
    # Fill any NaN values with column means
    features = features.fillna(features.mean())
    
    return features

# Add this to ShortTermPredictionSystem or monkey patch it
ShortTermPredictionSystem.prepare_prediction_input = prepare_prediction_input

# Add a simple forecast generator function
def generate_forecast(self, city_name, input_data, days=7):
    """Generate a weather forecast based on input data"""
    # This is a simplified forecast generator
    # In a real system, you would use your trained models here
    
    # Get the most recent data
    recent_data = input_data.iloc[-7:].copy()
    
    # Calculate averages from recent data
    avg_tmax = recent_data['tmax'].mean()
    avg_tmin = recent_data['tmin'].mean()
    avg_tavg = recent_data['tavg'].mean()
    avg_prcp = recent_data['prcp'].mean()
    avg_wspd = recent_data['wspd'].mean()
    avg_pres = recent_data['pres'].mean()
    avg_humidity = recent_data['humidity'].mean()
    
    # Weather conditions based on precipitation
    conditions = [
        {"condition": "Sunny", "icon": "sun"},
        {"condition": "Partly Cloudy", "icon": "cloud-sun"},
        {"condition": "Cloudy", "icon": "cloud"},
        {"condition": "Light Rain", "icon": "cloud-rain"},
        {"condition": "Rain", "icon": "cloud-showers-heavy"},
        {"condition": "Thunderstorm", "icon": "bolt"}
    ]
    
    # Generate forecast for each day
    forecast = []
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        
        # Add some random variation
        tmax_var = avg_tmax + (random.random() * 4 - 2)  # ±2°C variation
        tmin_var = avg_tmin + (random.random() * 3 - 1.5)  # ±1.5°C variation
        tavg_var = (tmax_var + tmin_var) / 2
        
        # Ensure tmin <= tavg <= tmax
        tmin_var = min(tmin_var, tavg_var)
        tmax_var = max(tmax_var, tavg_var)
        
        # Precipitation with some randomness
        prcp_var = max(0, avg_prcp * (random.random() * 2))
        
        # Determine condition based on precipitation
        if prcp_var < 0.1:
            condition_index = 0  # Sunny
        elif prcp_var < 1:
            condition_index = 1  # Partly Cloudy
        elif prcp_var < 3:
            condition_index = 2  # Cloudy
        elif prcp_var < 8:
            condition_index = 3  # Light Rain
        elif prcp_var < 15:
            condition_index = 4  # Rain
        else:
            condition_index = 5  # Thunderstorm
        
        # Create forecast entry
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "condition": conditions[condition_index]["condition"],
            "icon": conditions[condition_index]["icon"],
            "temp_avg": round(tavg_var, 1),
            "temp_min": round(tmin_var, 1),
            "temp_max": round(tmax_var, 1),
            "humidity": round(avg_humidity + (random.random() * 10 - 5)),
            "wind_speed": round(avg_wspd + (random.random() * 4 - 2), 1),
            "pressure": round(avg_pres + (random.random() * 4 - 2)),
            "precipitation": round(prcp_var, 1)
        })
    
    return forecast

# Monkey patch the generate_forecast method
ShortTermPredictionSystem.generate_forecast = generate_forecast

def main():
    """Main function to run the weather prediction system"""
    print("Starting Weather Prediction System...")
    
    # Create the main prediction system
    prediction_system = WeatherPredictionSystem()
    
    # Create and initialize the short-term prediction system
    short_term_system = ShortTermPredictionSystem(prediction_system)
    
    # Default city
    city = "Bangalore"
    
    # Check if a city was provided as a command-line argument
    if len(sys.argv) > 1 and sys.argv[1] != "do":  # Skip "do" from your command
        requested_city = sys.argv[1]
        if requested_city in prediction_system.cities:
            city = requested_city
        else:
            print(f"City '{requested_city}' not found. Using default city: {city}")
            print(f"Available cities: {', '.join(prediction_system.cities.keys())}")
    
    # Initialize the system for the specific city
    short_term_system.initialize_system(city)
    
    # Add the save_forecast_to_csv method if it doesn't exist
    if not hasattr(short_term_system, 'save_forecast_to_csv'):
        def save_forecast_to_csv(self, city_name, forecast, fallback=False):
            """Save the forecast to a CSV file"""
            try:
                # Check if forecast is empty or None
                if not forecast:
                    print("Warning: Empty forecast data, cannot save to CSV")
                    return
                
                # Convert forecast to DataFrame
                forecast_data = []
                for day in forecast:
                    forecast_data.append({
                        'date': day['date'],
                        'day': day['day'],
                        'condition': day['condition'],
                        'temp_avg': day['temp_avg'],
                        'temp_min': day['temp_min'],
                        'temp_max': day['temp_max'],
                        'humidity': day['humidity'],
                        'wind_speed': day['wind_speed'],
                        'pressure': day['pressure'],
                        'precipitation': day['precipitation']
                    })
                
                # Create DataFrame
                df = pd.DataFrame(forecast_data)
                
                # Save to CSV
                csv_path = os.path.join(self.prediction_system.output_dir, f"{city_name.lower()}_short_term_forecast.csv")
                df.to_csv(csv_path, index=False)
                
                if fallback:
                    print(f"Fallback forecast saved to {csv_path}")
                else:
                    print(f"Short-term forecast saved to {csv_path}")
                    
                # Print the first few rows to verify data is being saved
                print("\nForecast data preview:")
                print(df.head(3).to_string())
                
            except Exception as e:
                print(f"Error saving forecast to CSV: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Monkey patch the save_forecast_to_csv method
        ShortTermPredictionSystem.save_forecast_to_csv = save_forecast_to_csv
    
    # Run the prediction
    predictions = short_term_system.predict_short_term(city)
    
    # Save the forecast to CSV
    if predictions and len(predictions) > 0:
        short_term_system.save_forecast_to_csv(city, predictions)
        
        print("\nShort-term 7-day weather forecast for Bangalore:")
        print("=" * 80)
        
        # Display the forecast in a readable format
        for day in predictions:
            print(f"{day['day']} ({day['date']}): {day['condition']}, {day['temp_min']}°C to {day['temp_max']}°C, "
                  f"Precipitation: {day['precipitation']}mm, Humidity: {day['humidity']}%, "
                  f"Wind: {day['wind_speed']}m/s")
        
        print("=" * 80)
    else:
        print("\nNo forecast data was generated. Please check the model.")
    
    print("\nWeather prediction completed.")

if __name__ == "__main__":
    main()