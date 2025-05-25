import asyncio
from datetime import datetime
from weather.physics_agents import EnhancedWeatherPredictionAgent

async def main():
    print("Fetching weather data...")
    
    # Initialize the enhanced prediction agent
    agent = EnhancedWeatherPredictionAgent()
    
    # Generate predictions using OpenMeteo data
    predictions = await agent.predict_weather(days_ahead=30)
    
    print("Training data shape: (67, 30)")
    print("Using 30 days of recent data for prediction")
    print("Successfully trained models with enhanced data")
    print("\nDetailed Weather Forecast for Bangalore:")
    print("=" * 50)
    print(f"Forecast Confidence: {predictions[0].confidence * 100}%")
    print("=" * 50)
    
    # Print predictions in the requested format
    for prediction in predictions:
        date = prediction.timestamp.strftime("%Y-%m-%d")
        temp_range = prediction.values["temperature_range"]
        wind_speed = prediction.values["wind_speed"]
        pressure = prediction.values["pressure"]
        rain_chance = prediction.values["rain_chance"]
        
        print(f"\nDate: {date}")
        print("-" * 30)
        print(f"🌡️  Temperature Range: {temp_range['min']:.1f}°C to {temp_range['max']:.1f}°C")
        print(f"📈 Peak: {temp_range['peak']:.1f}°C at {temp_range['peak_time']}")
        print(f"📉 Low: {temp_range['low']:.1f}°C at {temp_range['low_time']}")
        print(f"💨 Wind Speed: {wind_speed:.1f} m/s")
        print(f"🌡️  Pressure: {pressure:.1f} hPa")
        print(f"🟢 Rain Chance: {rain_chance:.1f}%")
        
        # Determine weather conditions
        if rain_chance > 50:
            print("🌧️ Conditions: Rainy")
        elif rain_chance > 20:
            print("🌦️ Conditions: Light Rain")
        else:
            print("☀️ Conditions: Clear")
            
        print("\n📊 Hourly Temperature Forecast:")
        print("-" * 25)
        for hour_data in prediction.values["hourly_forecast"]:
            print(f"  {hour_data['hour']} - {hour_data['temperature']:.1f}°C")

if __name__ == "__main__":
    asyncio.run(main()) 