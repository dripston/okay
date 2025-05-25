import asyncio
from physics_agents import PressureGradientAgent, TemperatureAgent, PrecipitationAgent
from ensemble_predictor import WeatherEnsemble
from data_normalizer import WeatherDataNormalizer

async def test_ensemble_prediction():
    # Initialize physics agents
    pressure_agent = PressureGradientAgent()
    temperature_agent = TemperatureAgent()
    precipitation_agent = PrecipitationAgent()
    
    # Initialize ensemble and normalizer
    ensemble = WeatherEnsemble(input_size=4)  # pressure, temp, precip, confidence
    normalizer = WeatherDataNormalizer()
    
    print("\n=== Running Physics-ML Ensemble Test ===")
    
    # Get some initial physics predictions
    test_data = {
        "metrics": {
            "pressure": 1013.25,
            "temperature": 25.0,
            "wind_speed": 5.0,
            "cloud_cover": 30,
            "precipitation": 0.1,
            "humidity": 65,
            "radiation": 750
        }
    }
    
    # Get physics-based predictions
    pressure_state = await pressure_agent.compute_pressure_field(test_data)
    temperature_state = await temperature_agent.compute_temperature_field(test_data)
    precipitation_state = await precipitation_agent.compute_precipitation_field(test_data)
    
    physics_states = [pressure_state, temperature_state, precipitation_state]
    
    # Get ensemble predictions
    predictions, uncertainty = ensemble.forward(physics_states)
    
    print("\n=== Physics Agent Outputs ===")
    print(f"Pressure Gradient: {pressure_state.values['pressure_gradient']:.2f} hPa")
    print(f"Modified Temperature: {temperature_state.values['modified_temperature']:.2f}°C")
    print(f"Precipitation Rate: {precipitation_state.values['precipitation_rate']:.3f} mm/hr")
    
    print("\n=== Ensemble Model Predictions ===")
    print(f"Predicted Pressure: {predictions['pressure']:.2f} hPa")
    print(f"Predicted Temperature: {predictions['temperature']:.2f}°C")
    print(f"Predicted Precipitation: {predictions['precipitation']:.3f} mm/hr")
    print(f"Prediction Uncertainty: {uncertainty:.3f}")

if __name__ == "__main__":
    asyncio.run(test_ensemble_prediction())