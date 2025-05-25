import asyncio
from physics_agents import (
    PressureGradientAgent, 
    TemperatureAgent, 
    PrecipitationAgent,
    OpenMeteoDataFetcher,
    EnhancedWeatherPredictionAgent
)
from data_normalizer import WeatherDataNormalizer
from iot_stations import IoTStationManager
from satellite_feed import INSAT3DFeed
from crowdsource_data import CrowdsourceManager
import json
from datetime import datetime

async def test_physics_agents():
    # Initialize data sources
    iot_manager = IoTStationManager()
    satellite_feed = INSAT3DFeed()
    crowdsource_manager = CrowdsourceManager()
    normalizer = WeatherDataNormalizer()
    
    # Initialize physics agents
    pressure_agent = PressureGradientAgent()
    temperature_agent = TemperatureAgent()
    precipitation_agent = PrecipitationAgent()
    
    print("\n=== Fetching Base Weather Data ===")
    # Get normalized weather data
    iot_data = await iot_manager.fetch_all_stations()
    satellite_data = await satellite_feed.get_current_data()
    mobile_reports = await crowdsource_manager.simulate_mobile_reports(num_reports=50)
    crowdsource_data = await crowdsource_manager.process_reports(mobile_reports)
    
    normalized_iot = normalizer.normalize_iot_data(iot_data)
    normalized_satellite = normalizer.normalize_satellite_data(satellite_data)
    normalized_crowdsource = normalizer.normalize_crowdsource_data(crowdsource_data)
    
    all_data = normalized_iot + [normalized_satellite, normalized_crowdsource]
    base_weather = normalizer.resolve_conflicts(all_data)
    
    print("\n=== Running Physics Simulations ===")
    # Run physics computations
    pressure_state = await pressure_agent.compute_pressure_field(base_weather)
    temperature_state = await temperature_agent.compute_temperature_field(base_weather)
    precipitation_state = await precipitation_agent.compute_precipitation_field(base_weather)
    
    # Publish states to Redis
    await pressure_agent.publish_state(pressure_state)
    await temperature_agent.publish_state(temperature_state)
    await precipitation_agent.publish_state(precipitation_state)
    
    print("\n=== Physics-Enhanced Weather Data ===")
    print("Pressure Analysis:", json.dumps(pressure_state.values, indent=2))
    print("Temperature Analysis:", json.dumps(temperature_state.values, indent=2))
    print("Precipitation Analysis:", json.dumps(precipitation_state.values, indent=2))
    
    # Get latest states from Redis to verify communication
    print("\n=== Verifying Redis Communication ===")
    latest_states = await pressure_agent.get_latest_states()
    print(f"Number of states in Redis: {len(latest_states)}")
    for state in latest_states[-3:]:  # Show last 3 states
        print(f"Agent State: {state.values}")

async def test_weather_prediction():
    print("\n=== Testing Weather Prediction with OpenMeteo ===")
    
    # Initialize the enhanced prediction agent
    agent = EnhancedWeatherPredictionAgent()
    
    # Generate predictions using OpenMeteo data
    predictions = await agent.predict_weather(days_ahead=30)
    
    print("\n=== Weather Forecast Results ===")
    print(f"Forecast Confidence: {predictions[0].confidence * 100}%")
    print("=" * 50)
    
    # Print predictions for the next 5 days
    for prediction in predictions[:5]:
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

async def main():
    await test_physics_agents()
    await test_weather_prediction()

if __name__ == "__main__":
    asyncio.run(main())