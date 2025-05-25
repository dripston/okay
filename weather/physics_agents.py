import numpy as np
import redis
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import aiohttp
import pandas as pd

@dataclass
class PhysicsState:
    timestamp: datetime
    location: Dict[str, float]
    values: Dict[str, float]
    confidence: float

class BasePhysicsAgent:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.stream_key = "weather_physics"
        
    async def publish_state(self, state: PhysicsState):
        """Publish agent state to Redis using SET instead of XADD"""
        data = {
            "timestamp": state.timestamp.isoformat(),
            "location": json.dumps(state.location),
            "values": json.dumps(state.values),
            "confidence": state.confidence,
            "agent_type": self.__class__.__name__
        }
        key = f"{self.stream_key}:{self.__class__.__name__}:{state.timestamp.isoformat()}"
        self.redis_client.set(key, json.dumps(data))
        
    async def get_latest_states(self) -> List[PhysicsState]:
        """Get latest states using key pattern matching"""
        keys = self.redis_client.keys(f"{self.stream_key}:*")
        states = []
        for key in keys:
            data = json.loads(self.redis_client.get(key))
            states.append(PhysicsState(
                timestamp=datetime.fromisoformat(data['timestamp']),
                location=json.loads(data['location']),
                values=json.loads(data['values']),
                confidence=float(data['confidence'])
            ))
        return states

class PressureGradientAgent(BasePhysicsAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.viscosity = 1.825e-5  # Air kinematic viscosity
        self.density = 1.225       # Air density at sea level
        
    async def compute_pressure_field(self, normalized_data: Dict) -> PhysicsState:
        """Compute pressure gradients using simplified Navier-Stokes"""
        metrics = normalized_data["metrics"]
        
        # Simplified pressure gradient calculation
        pressure_base = metrics["pressure"] if metrics["pressure"] else 1013.25
        temperature = metrics["temperature"]
        wind_speed = metrics["wind_speed"] if metrics["wind_speed"] else 0
        
        # Calculate pressure variations using Bernoulli's principle
        dynamic_pressure = 0.5 * self.density * (wind_speed ** 2)
        pressure_gradient = self._compute_gradient(pressure_base, dynamic_pressure, temperature)
        
        return PhysicsState(
            timestamp=datetime.now(),
            location={"latitude": 12.9716, "longitude": 77.5946},
            values={"pressure_gradient": pressure_gradient},
            confidence=0.85
        )
        
    def _compute_gradient(self, base_pressure: float, dynamic_pressure: float, 
                         temperature: float) -> float:
        """Compute pressure gradient considering temperature effects"""
        return base_pressure + dynamic_pressure * (1 - (temperature - 25) * 0.002)

class TemperatureAgent(BasePhysicsAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.building_height = 30  # Average building height in meters
        self.street_width = 20     # Average street width in meters
        
    async def compute_temperature_field(self, normalized_data: Dict) -> PhysicsState:
        """Compute temperature distribution with urban canyon effects"""
        metrics = normalized_data["metrics"]
        base_temp = metrics["temperature"]
        radiation = metrics["radiation"] if metrics["radiation"] else 800
        
        # Urban canyon temperature modification
        sky_view_factor = self._compute_sky_view_factor()
        urban_temp_delta = self._compute_urban_heat_island(radiation, sky_view_factor)
        
        modified_temp = base_temp + urban_temp_delta
        
        return PhysicsState(
            timestamp=datetime.now(),
            location={"latitude": 12.9716, "longitude": 77.5946},
            values={"modified_temperature": modified_temp},
            confidence=0.82
        )
        
    def _compute_sky_view_factor(self) -> float:
        """Compute sky view factor for urban canyon"""
        aspect_ratio = self.building_height / self.street_width
        aspect_ratio = min(max(aspect_ratio, -1), 1)
        return np.arccos(aspect_ratio) / np.pi
        
    def _compute_urban_heat_island(self, radiation: float, sky_view_factor: float) -> float:
        """Compute urban heat island effect"""
        return (1 - sky_view_factor) * (radiation / 1000) * 2

class PrecipitationAgent(BasePhysicsAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.autoconversion_threshold = 0.5  # g/kg
        self.collection_efficiency = 0.95
        
    async def compute_precipitation_field(self, normalized_data: Dict) -> PhysicsState:
        """Compute precipitation using modified Kessler scheme"""
        metrics = normalized_data["metrics"]
        
        cloud_water = metrics["cloud_cover"] / 100 if metrics["cloud_cover"] else 0
        current_precip = metrics["precipitation"] if metrics["precipitation"] else 0
        humidity = metrics["humidity"] if metrics["humidity"] else 60
        
        # Modified Kessler microphysics
        autoconversion = self._compute_autoconversion(cloud_water)
        collection = self._compute_collection(cloud_water, current_precip)
        evaporation = self._compute_evaporation(current_precip, humidity)
        
        net_precipitation = max(0, autoconversion + collection - evaporation)
        
        return PhysicsState(
            timestamp=datetime.now(),
            location={"latitude": 12.9716, "longitude": 77.5946},
            values={"precipitation_rate": net_precipitation},
            confidence=0.78
        )
        
    def _compute_autoconversion(self, cloud_water: float) -> float:
        """Compute autoconversion of cloud water to rain"""
        if cloud_water > self.autoconversion_threshold:
            return 0.001 * (cloud_water - self.autoconversion_threshold)
        return 0
        
    def _compute_collection(self, cloud_water: float, current_precip: float) -> float:
        """Compute collection of cloud water by rain"""
        return self.collection_efficiency * cloud_water * current_precip * 0.1
        
    def _compute_evaporation(self, precip: float, humidity: float) -> float:
        """Compute evaporation of precipitation"""
        return precip * (1 - humidity/100) * 0.1

class WeatherPredictionAgent(BasePhysicsAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thermal_diffusivity = 2.0e-5  # m²/s
        self.specific_heat = 1005  # J/kg·K
        self.latent_heat = 2.5e6  # J/kg
        self.gas_constant = 287.05  # J/kg·K
        self.gravity = 9.81  # m/s²
        self.density = 1.225  # Air density at sea level
        
    async def predict_weather(self, historical_data: Dict, days_ahead: int = 30) -> List[PhysicsState]:
        """Generate detailed weather forecast using physics-based modeling"""
        predictions = []
        current_date = datetime.now()
        
        # Extract base metrics from historical data
        base_temp = historical_data.get("temperature", 25.0)
        base_pressure = historical_data.get("pressure", 1013.25)
        base_humidity = historical_data.get("humidity", 60.0)
        base_wind = historical_data.get("wind_speed", 5.0)
        
        for day in range(days_ahead):
            date = current_date + timedelta(days=day)
            
            # Calculate diurnal temperature variation
            daily_temps = self._compute_diurnal_temperature(base_temp, day)
            
            # Calculate pressure changes
            pressure = self._compute_pressure(base_pressure, daily_temps, base_wind)
            
            # Calculate precipitation probability
            rain_chance = self._compute_precipitation_probability(base_humidity, pressure)
            
            # Calculate wind speed variations
            wind_speed = self._compute_wind_speed(base_wind, pressure)
            
            # Generate hourly predictions
            hourly_predictions = []
            for hour in range(0, 24, 3):  # Every 3 hours
                temp = daily_temps[hour//3]
                hourly_predictions.append({
                    "hour": f"{hour:02d}:00",
                    "temperature": temp,
                    "wind_speed": wind_speed,
                    "pressure": pressure,
                    "rain_chance": rain_chance
                })
            
            # Create prediction state
            prediction = PhysicsState(
                timestamp=date,
                location={"latitude": 12.9716, "longitude": 77.5946},  # Bangalore coordinates
                values={
                    "temperature_range": {
                        "min": min(daily_temps),
                        "max": max(daily_temps),
                        "peak": max(daily_temps),
                        "peak_time": "15:00",
                        "low": min(daily_temps),
                        "low_time": "06:00"
                    },
                    "wind_speed": wind_speed,
                    "pressure": pressure,
                    "rain_chance": rain_chance,
                    "hourly_forecast": hourly_predictions
                },
                confidence=0.75  # Base confidence level
            )
            predictions.append(prediction)
            
        return predictions
    
    def _compute_diurnal_temperature(self, base_temp: float, day: int) -> List[float]:
        """Compute diurnal temperature variations using heat diffusion equation"""
        temps = []
        amplitude = 5.0 + (day * 0.1)  # Increasing amplitude as summer approaches
        phase_shift = 0.0
        
        for hour in range(0, 24, 3):
            # Simplified heat diffusion equation
            time = hour / 24.0
            temp = base_temp + amplitude * np.sin(2 * np.pi * time + phase_shift)
            temps.append(round(temp, 1))
            
        return temps
    
    def _compute_pressure(self, base_pressure: float, temps: List[float], wind_speed: float) -> float:
        """Compute pressure using ideal gas law and Bernoulli's principle"""
        avg_temp = sum(temps) / len(temps)
        dynamic_pressure = 0.5 * self.density * (wind_speed ** 2)
        
        # Pressure variation with temperature
        pressure = base_pressure * (avg_temp + 273.15) / (25 + 273.15)
        
        # Add dynamic pressure component
        pressure += dynamic_pressure
        
        return round(pressure, 1)
    
    def _compute_precipitation_probability(self, humidity: float, pressure: float) -> float:
        """Compute precipitation probability using relative humidity and pressure"""
        # Base probability from humidity
        base_prob = (humidity - 50) / 50.0 if humidity > 50 else 0
        
        # Pressure influence
        pressure_factor = (pressure - 1013.25) / 1013.25
        
        # Combine factors
        probability = base_prob * (1 - pressure_factor)
        
        return round(max(0, min(100, probability * 100)), 1)
    
    def _compute_wind_speed(self, base_wind: float, pressure: float) -> float:
        """Compute wind speed using pressure gradient force"""
        # Simplified geostrophic wind approximation
        pressure_gradient = (pressure - 1013.25) / 1013.25
        wind_speed = base_wind * (1 + pressure_gradient)
        
        return round(max(0, wind_speed), 1)

class OpenMeteoDataFetcher:
    def __init__(self, latitude: float = 12.9716, longitude: float = 77.5946):
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        
    async def fetch_historical_data(self, days: int = 30) -> Dict:
        """Fetch historical weather data from OpenMeteo"""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,precipitation",
            "past_days": days,
            "timezone": "auto"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_historical_data(data)
                else:
                    raise Exception(f"Failed to fetch data: {response.status}")
    
    def _process_historical_data(self, data: Dict) -> Dict:
        """Process raw OpenMeteo data into format needed for physics models"""
        hourly = data.get("hourly", {})
        
        # Convert to pandas DataFrame for easier processing
        df = pd.DataFrame({
            "temperature": hourly.get("temperature_2m", []),
            "humidity": hourly.get("relative_humidity_2m", []),
            "pressure": hourly.get("pressure_msl", []),
            "wind_speed": hourly.get("wind_speed_10m", []),
            "precipitation": hourly.get("precipitation", [])
        })
        
        # Calculate daily averages and trends
        processed_data = {
            "temperature": df["temperature"].mean(),
            "humidity": df["humidity"].mean(),
            "pressure": df["pressure"].mean(),
            "wind_speed": df["wind_speed"].mean(),
            "precipitation": df["precipitation"].sum(),
            "temperature_trend": df["temperature"].pct_change().mean(),
            "pressure_trend": df["pressure"].pct_change().mean(),
            "wind_trend": df["wind_speed"].pct_change().mean()
        }
        
        return processed_data

class EnhancedWeatherPredictionAgent(WeatherPredictionAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_fetcher = OpenMeteoDataFetcher()
        
    async def predict_weather(self, days_ahead: int = 30) -> List[PhysicsState]:
        """Generate enhanced weather predictions using OpenMeteo data"""
        print("Fetching weather data...")
        historical_data = await self.data_fetcher.fetch_historical_data()
        print("Data fetched successfully")
        print("Successfully loaded historical data")
        
        # Get base predictions from parent class
        base_predictions = await super().predict_weather(historical_data, days_ahead)
        
        # Enhance predictions with OpenMeteo trends
        for prediction in base_predictions:
            # Adjust temperature based on trend
            temp_trend = historical_data["temperature_trend"]
            prediction.values["temperature_range"]["min"] *= (1 + temp_trend)
            prediction.values["temperature_range"]["max"] *= (1 + temp_trend)
            
            # Adjust pressure based on trend
            pressure_trend = historical_data["pressure_trend"]
            prediction.values["pressure"] *= (1 + pressure_trend)
            
            # Adjust wind speed based on trend
            wind_trend = historical_data["wind_trend"]
            prediction.values["wind_speed"] *= (1 + wind_trend)
            
            # Update hourly forecasts
            for hour_data in prediction.values["hourly_forecast"]:
                hour_data["temperature"] *= (1 + temp_trend)
                hour_data["wind_speed"] *= (1 + wind_trend)
                hour_data["pressure"] *= (1 + pressure_trend)
        
        return base_predictions