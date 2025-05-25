from datetime import datetime
import aiohttp
import asyncio
from typing import Dict, Optional
import numpy as np

class INSAT3DFeed:
    def __init__(self):
        self.base_url = "https://satellite.imd.gov.in/insat3d"  # Example URL
        self.bangalore_bounds = {
            "lat_min": 12.7342,
            "lat_max": 13.1393,
            "lon_min": 77.3579,
            "lon_max": 77.8567
        }
        
    async def fetch_satellite_data(self) -> Dict:
        """
        Fetch INSAT-3D satellite data for Bangalore region
        Currently generating synthetic data to simulate satellite feed
        """
        # Simulate satellite data retrieval delay
        await asyncio.sleep(0.2)
        
        # Create grid points for Bangalore region
        lat_points = np.linspace(
            self.bangalore_bounds["lat_min"],
            self.bangalore_bounds["lat_max"],
            20
        )
        lon_points = np.linspace(
            self.bangalore_bounds["lon_min"],
            self.bangalore_bounds["lon_max"],
            20
        )
        
        # Generate satellite data grid
        grid_data = []
        for lat in lat_points:
            for lon in lon_points:
                grid_data.append({
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "temperature": round(25 + np.random.normal(0, 2), 2),
                    "cloud_cover": round(np.random.uniform(0, 100), 2),
                    "precipitation": round(max(0, np.random.normal(2, 5)), 2),
                    "radiation": round(np.random.uniform(600, 1000), 2)
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "source": "INSAT-3D",
            "resolution": "1km",
            "grid_data": grid_data,
            "metadata": {
                "satellite": "INSAT-3D",
                "channels": ["VIS", "SWIR", "MIR", "TIR1", "TIR2"],
                "coverage": self.bangalore_bounds
            }
        }

    def process_raw_data(self, raw_data: Dict) -> Dict:
        processed_data = {
            "timestamp": raw_data["timestamp"],
            "source": raw_data["source"],
            "aggregated_data": {
                "average_temperature": round(float(np.mean([point["temperature"] for point in raw_data["grid_data"]])), 2),
                "average_cloud_cover": round(float(np.mean([point["cloud_cover"] for point in raw_data["grid_data"]])), 2),
                "average_precipitation": round(float(np.mean([point["precipitation"] for point in raw_data["grid_data"]])), 2),
                "max_radiation": round(float(max([point["radiation"] for point in raw_data["grid_data"]])), 2)
            },
            "grid_resolution": raw_data["resolution"]
        }
        return processed_data

    async def get_current_data(self) -> Dict:
        """
        Get and process current satellite data
        """
        raw_data = await self.fetch_satellite_data()
        return self.process_raw_data(raw_data)