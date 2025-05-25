from dataclasses import dataclass
from datetime import datetime
import aiohttp
import asyncio
from typing import Dict, List
import random  # For demo data generation

@dataclass
class IoTStation:
    station_id: str
    name: str
    latitude: float
    longitude: float
    area: str
    api_endpoint: str

class IoTStationManager:
    def __init__(self):
        self.stations = self._initialize_stations()
    
    def _initialize_stations(self) -> Dict[str, IoTStation]:
        # Bangalore areas for station distribution
        areas = [
            ("Central", 12.9716, 77.5946),  # City Center
            ("Electronic City", 12.8399, 77.6770),
            ("Whitefield", 12.9698, 77.7500),
            ("Marathahalli", 12.9591, 77.6974),
            ("Hebbal", 12.9987, 77.5967),
            ("Bannerghatta", 12.8625, 77.5960),
            ("Yelahanka", 13.1005, 77.5960),
            ("JP Nagar", 12.9107, 77.5922),
            ("Koramangala", 12.9352, 77.6245),
            ("Indiranagar", 12.9784, 77.6408)
        ]
        
        stations = {}
        # Create 5 stations in each area for total 50 stations
        for area, base_lat, base_lon in areas:
            for i in range(5):
                # Add small offsets to distribute stations within each area
                lat_offset = random.uniform(-0.01, 0.01)
                lon_offset = random.uniform(-0.01, 0.01)
                
                station_id = f"{area.lower().replace(' ', '_')}_{i+1}"
                stations[station_id] = IoTStation(
                    station_id=station_id,
                    name=f"{area} Station {i+1}",
                    latitude=base_lat + lat_offset,
                    longitude=base_lon + lon_offset,
                    area=area,
                    api_endpoint=f"https://api.weatherstation.com/stations/{station_id}"
                )
        
        return stations

    async def fetch_station_data(self, station: IoTStation) -> Dict:
        """
        Fetch data from a single IoT station
        For demo, generating synthetic data based on location and time
        """
        # Simulate network request
        await asyncio.sleep(0.1)
        
        # Generate realistic weather data based on location and season
        base_temp = 25 + random.uniform(-5, 5)
        humidity = 60 + random.uniform(-20, 20)
        pressure = 1013 + random.uniform(-10, 10)
        wind_speed = random.uniform(0, 15)
        
        return {
            "station_id": station.station_id,
            
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": round(base_temp, 2),
            "humidity": round(humidity, 2),
            "pressure": round(pressure, 2),
            "wind_speed": round(wind_speed, 2),
            "location": {
                "latitude": round(station.latitude, 4),
                "longitude": round(station.longitude, 4)
            }
        }

    async def fetch_all_stations(self) -> List[Dict]:
        """Fetch data from all stations concurrently"""
        tasks = [
            self.fetch_station_data(station)
            for station in self.stations.values()
        ]
        return await asyncio.gather(*tasks)