import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GridPoint:
    latitude: float
    longitude: float
    altitude: float
    values: Dict[str, float]


@dataclass
class TimeStep:
    timestamp: datetime
    grid_points: List[GridPoint]


class BaseWeatherClient:
    """Abstract base class for weather clients"""
    async def get_current_weather(self, lat: float, lon: float) -> Dict:
        raise NotImplementedError

    async def get_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict:
        raise NotImplementedError


class OpenMeteoClient(BaseWeatherClient):
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1"
        self.session = aiohttp.ClientSession()
        
    async def get_current_weather(self, lat: float, lon: float) -> Dict:
        """Get current weather data from OpenMeteo"""
        url = f"{self.base_url}/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation",
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation",
            "timezone": "auto"
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to fetch data from OpenMeteo: {response.status}")
                
    async def get_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict:
        """Get historical weather data from OpenMeteo"""
        url = f"{self.base_url}/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,precipitation",
            "timezone": "auto"
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to fetch historical data from OpenMeteo: {response.status}")


class NWPSClient(BaseWeatherClient):
    """Client for NWPS weather forecast service. This is designed to be almost a drop-in replacement for OpenMeteo."""
    def __init__(self):
        # This URL is a placeholder. Replace with the actual NWPS API endpoint.
        self.base_url = "https://api.nwps.example.com/v1"
        self.session = aiohttp.ClientSession()
        
    async def get_current_weather(self, lat: float, lon: float) -> Dict:
        url = f"{self.base_url}/forecast"
        # NWPS might have different parameter naming; adjust as needed
        params = {
            "lat": lat,
            "lon": lon,
            "data": "temp,humidity,pressure,wind,u_wind,v_wind,precip",
            "timezone": "auto"
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                # Assuming NWPS returns data in a similar format.
                return await response.json()
            else:
                raise Exception(f"Failed to fetch data from NWPS: {response.status}")
                
    async def get_historical_weather(self, lat: float, lon: float, start_date: str, end_date: str) -> Dict:
        """Get historical weather data from NWPS"""
        url = f"{self.base_url}/archive"
        params = {
            "lat": lat,
            "lon": lon,
            "start": start_date,
            "end": end_date,
            "data": "temp,humidity,pressure,wind,u_wind,v_wind,precip",
            "timezone": "auto"
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Failed to fetch historical data from NWPS: {response.status}")


class WeatherProcessor:
    def __init__(self, weather_client: Optional[BaseWeatherClient] = None):
        # If no client is provided, default to OpenMeteo
        self.meteo_client = weather_client if weather_client else OpenMeteoClient()
        self.temp_bounds = (273.15, 313.15)  # 0°C to 40°C in Kelvin
        self.wind_bounds = (0.0, 30.0)  # 0 to 30 m/s
        self.humidity_bounds = (0.0, 1.0)  # 0 to 100% relative humidity as fraction
        self.pressure_bounds = (950.0, 1050.0)  # hPa
        
    def _convert_to_grid_points(self, data: Dict, timestamp: datetime) -> List[GridPoint]:
        """Convert weather data to grid points. Supports both OpenMeteo and NWPS data formats."""
        grid_points = []
        
        # Handle differences in API response format
        if 'current' in data:  # Assuming OpenMeteo format
            current = data.get('current', {})
            lat = data.get('latitude', 0)
            lon = data.get('longitude', 0)
        else:
            # For NWPS, assume the keys differ (adjust as needed)
            current = data
            lat = data.get('lat', 0)
            lon = data.get('lon', 0)
            # NWPS might provide temperature in Celsius; convert to Kelvin if necessary
            if 'temp' in current and current['temp'] < 200:  
                current['temp'] = current['temp'] + 273.15
            # Map NWPS field names to expected names
            # This mapping may need adjustments based on actual API specifications
            current.setdefault('temperature_2m', current.get('temp', 0))
            current.setdefault('relative_humidity_2m', current.get('humidity', 50))
            current.setdefault('pressure_msl', current.get('pressure', 1013.25))
            current.setdefault('wind_speed_10m', current.get('wind', 0))
            # For wind direction, NWPS may provide u_wind and v_wind separately
            if 'u_wind' in current and 'v_wind' in current and current['u_wind'] is not None and current['v_wind'] is not None:
                # Derive wind speed and direction from components
                u = current['u_wind']
                v = current['v_wind']
                current['wind_speed_10m'] = np.sqrt(u*u + v*v)
                current['wind_direction_10m'] = np.degrees(np.arctan2(-u, -v)) % 360
            else:
                current.setdefault('wind_direction_10m', 0)
            current.setdefault('precipitation', current.get('precip', 0))
            # Rename key to match the expected naming by subsequent functions
            current = current
        
        # Convert wind direction to u,v components
        wind_speed = current.get('wind_speed_10m', 0)
        wind_dir = np.radians(current.get('wind_direction_10m', 0))
        u_wind = -wind_speed * np.sin(wind_dir)
        v_wind = -wind_speed * np.cos(wind_dir)
        
        values = {
            'temperature': current.get('temperature_2m', 0),  # Kelvin
            'pressure': current.get('pressure_msl', 1013.25),
            'specific_humidity': self._rh_to_specific_humidity(
                current.get('relative_humidity_2m', 50),
                current.get('temperature_2m', 0),
                current.get('pressure_msl', 1013.25)
            ),
            'u_wind': u_wind,
            'v_wind': v_wind,
            'wind_speed': wind_speed,
            'wind_direction': current.get('wind_direction_10m', 0),
            'precipitation': current.get('precipitation', 0),
            'relative_humidity': current.get('relative_humidity_2m', 50)
        }
        
        grid_points.append(GridPoint(
            latitude=lat,
            longitude=lon,
            altitude=0.0,
            values=values
        ))
        
        return grid_points
        
    def _rh_to_specific_humidity(self, rh: float, T: float, p: float) -> float:
        """Convert relative humidity to specific humidity"""
        # Saturation vapor pressure (hPa)
        es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
        
        # Vapor pressure (hPa)
        e = es * rh / 100
        
        # Specific humidity (kg/kg)
        return 0.622 * e / (p - 0.378 * e)
        
    def _apply_physics_enhancements(self, grid_points: List[GridPoint]):
        """Apply physics-based enhancements to the weather data"""
        for point in grid_points:
            # Get current values
            T = point.values['temperature']
            p = point.values['pressure']
            q = point.values['specific_humidity']
            u = point.values['u_wind']
            v = point.values['v_wind']
            
            # Apply thermal wind effect with reduced impact
            lat_rad = np.radians(point.latitude)
            f = 2 * 7.2921e-5 * np.sin(lat_rad)  # Coriolis parameter
            
            # Temperature gradient effect (reduced)
            dT_dy = 0.01  # K/km (reduced from 0.1)
            thermal_wind = 0.1 * 9.81 / (f * T) * dT_dy  # Added 0.1 factor to reduce impact
            
            # Update wind components with damping
            point.values['u_wind'] = np.clip(
                u * 0.8 + thermal_wind,  # Damp original wind by 20%
                *self.wind_bounds
            )
            point.values['v_wind'] = np.clip(
                v * 0.8 + thermal_wind,
                *self.wind_bounds
            )
            
            # Add surface friction effect
            wind_speed = np.sqrt(
                point.values['u_wind']**2 + point.values['v_wind']**2
            )
            friction_factor = 1 - 0.1 * wind_speed  # Increase friction with wind speed
            point.values['u_wind'] *= friction_factor
            point.values['v_wind'] *= friction_factor
            
            # Update wind speed and direction
            point.values['wind_speed'] = np.sqrt(
                point.values['u_wind']**2 + point.values['v_wind']**2
            )
            point.values['wind_direction'] = np.degrees(
                np.arctan2(-point.values['u_wind'], -point.values['v_wind'])
            ) % 360
            
            # Apply moisture adjustments with reduced sensitivity
            es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
            qs = 0.622 * es / p
            
            # Update specific humidity with reduced temperature dependence
            point.values['specific_humidity'] = np.clip(
                q * (1 + 0.05 * (T - 288.15) / 10),  # Reduced from 0.1 to 0.05
                *self.humidity_bounds
            )
            
            # Update relative humidity with proper bounds
            point.values['relative_humidity'] = np.clip(
                point.values['specific_humidity'] / qs * 100,
                0, 100  # Ensure RH stays between 0 and 100%
            )
            
            # Add diurnal variation to temperature
            hour = datetime.now().hour
            diurnal_factor = 1 + 0.1 * np.sin(2 * np.pi * (hour - 14) / 24)  # Peak at 2 PM
            point.values['temperature'] = np.clip(
                T * diurnal_factor,
                *self.temp_bounds
            )
            
            # Add pressure adjustment based on temperature
            point.values['pressure'] = np.clip(
                p * (1 - 0.0001 * (T - 288.15)),  # Small pressure adjustment
                *self.pressure_bounds
            )
            
    async def process_weather_data(self, lat: float, lon: float) -> Dict:
        """Process weather data with physics enhancements"""
        # Get current weather data
        data = await self.meteo_client.get_current_weather(lat, lon)
        
        # Convert to grid points
        grid_points = self._convert_to_grid_points(data, datetime.now())
        
        # Apply physics enhancements
        self._apply_physics_enhancements(grid_points)
        
        # Convert back to standard format (convert temperature to Celsius)
        result = {
            'temperature': grid_points[0].values['temperature'] - 273.15,
            'pressure': grid_points[0].values['pressure'],
            'humidity': grid_points[0].values['relative_humidity'],
            'wind_speed': grid_points[0].values['wind_speed'],
            'wind_direction': grid_points[0].values['wind_direction'],
            'precipitation': grid_points[0].values['precipitation']
        }
        
        return result


async def main():
    # Example coordinates (Bangalore)
    lat = 12.9716
    lon = 77.5946

    # Check command line arguments to choose client
    client_type = 'openmeteo'
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'nwps':
        client_type = 'nwps'

    if client_type == 'nwps':
        logger.info("Using NWPSClient for weather data.")
        weather_client = NWPSClient()
    else:
        logger.info("Using OpenMeteoClient for weather data.")
        weather_client = OpenMeteoClient()

    processor = WeatherProcessor(weather_client)

    try:
        result = await processor.process_weather_data(lat, lon)
        logger.info("Current weather with physics enhancements:")
        logger.info(f"Temperature: {result['temperature']:.1f}°C")
        logger.info(f"Pressure: {result['pressure']:.1f} hPa")
        logger.info(f"Humidity: {result['humidity']:.1f}%")
        logger.info(f"Wind Speed: {result['wind_speed']:.1f} m/s")
        logger.info(f"Wind Direction: {result['wind_direction']:.1f}°")
        logger.info(f"Precipitation: {result['precipitation']:.1f} mm/h")
    except Exception as e:
        logger.error(f"Error processing weather data: {e}")
    finally:
        # Close the session for whichever client was used
        await weather_client.session.close()


if __name__ == "__main__":
    asyncio.run(main())
