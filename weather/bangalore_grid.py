import numpy as np
import folium
import h3
from typing import Dict, List, Tuple
import requests
import json
from dataclasses import dataclass

@dataclass
class GridCell:
    h3_index: str
    center: Tuple[float, float]
    elevation: float
    urban_density: float
    
class BangaloreGrid:
    def __init__(self):
        # Bangalore's approximate center
        self.center = (12.9716, 77.5946)
        self.resolution = 9  # ~1km² hexagons
        self.cells: Dict[str, GridCell] = {}
        
        # Initialize grid
        self._create_grid()
        self._load_elevation_data()
        self._calculate_urban_density()
        
    def _create_grid(self):
        """Create hexagonal grid covering Bangalore"""
        # H3 expects (lat, lng) order
        base_hex = h3.latlng_to_cell(self.center[0], self.center[1], self.resolution)
        hex_indices = h3.grid_disk(base_hex, 15)
        
        for hex_id in hex_indices:
            # Reverse coordinates for cell_to_latlng as it returns (lat, lng)
            lat, lng = h3.cell_to_latlng(hex_id)
            self.cells[hex_id] = GridCell(
                h3_index=hex_id,
                center=(lat, lng),  # Store in consistent (lat, lng) order
                elevation=0.0,
                urban_density=0.0
            )
    
    def _load_elevation_data(self):
        """Load elevation data using local approximation"""
        # Bangalore elevation varies from ~840m to ~960m
        # Create elevation gradient based on distance from center
        for hex_id, cell in self.cells.items():
            lat, lng = cell.center
            # Calculate distance from center (normalized)
            dist_from_center = np.sqrt(
                (lat - self.center[0])**2 + (lng - self.center[1])**2
            ) * 100
            
            # Elevation varies with distance and direction
            # Higher elevations towards north and east
            lat_factor = (lat - self.center[0]) * 50
            lng_factor = (lng - self.center[1]) * 50
            
            # Base elevation (920m) + variation
            cell.elevation = 920 + lat_factor + lng_factor + dist_from_center
            # Clamp elevation to realistic range
            cell.elevation = max(840, min(960, cell.elevation))
    
    def _calculate_urban_density(self):
        """Calculate urban density using OpenStreetMap building data"""
        for hex_id, cell in self.cells.items():
            lat, lng = cell.center
            # Mock density based on distance from center
            dist_from_center = np.sqrt(
                (lat - self.center[0])**2 + (lng - self.center[1])**2
            )
            cell.urban_density = max(0, 1 - (dist_from_center * 10))
            
    def adjust_prediction(self, base_prediction: Dict[str, float], location: Tuple[float, float]) -> Dict[str, float]:
        """Adjust weather predictions based on microclimate factors"""
        hex_id = h3.latlng_to_cell(location[0], location[1], self.resolution)
        if hex_id not in self.cells:
            return base_prediction
            
        cell = self.cells[hex_id]
        
        # Adjust temperature based on urban heat island effect
        temp_adjustment = cell.urban_density * 2.0  # Up to 2°C warmer in dense areas
        elevation_adjustment = (920 - cell.elevation) * 0.0065  # Standard lapse rate
        
        adjusted = base_prediction.copy()
        if 'temperature' in adjusted:
            adjusted['temperature'] += temp_adjustment + elevation_adjustment
            
        # Adjust precipitation based on elevation
        if 'precipitation' in adjusted:
            if cell.elevation > 920:  # Above average elevation
                adjusted['precipitation'] *= 1.1  # 10% more precipitation
                
        return adjusted

    def create_visualization(self, predictions: Dict[str, Dict[str, float]], param: str) -> folium.Map:
        """Create a Leaflet map visualization for a weather parameter"""
        m = folium.Map(location=self.center, zoom_start=11)
        
        for hex_id, cell in self.cells.items():
            if hex_id in predictions:
                value = predictions[hex_id][param]
                # Update h3 boundary function call
                boundaries = h3.cell_to_boundary(hex_id)
                boundaries = [[lat, lng] for lat, lng in boundaries]
                
                # Color based on value
                color = self._get_color_for_value(value, param)
                
                folium.Polygon(
                    locations=boundaries,
                    color=color,
                    fill=True,
                    popup=f"{param}: {value:.2f}"
                ).add_to(m)
                
        return m
    
    def _get_color_for_value(self, value: float, param: str) -> str:
        """Get color for visualization based on parameter value"""
        if param == 'temperature':
            if value < 20: return 'blue'
            elif value < 25: return 'green'
            elif value < 30: return 'yellow'
            else: return 'red'
        elif param == 'precipitation':
            if value < 0.1: return 'white'
            elif value < 1.0: return 'lightblue'
            elif value < 5.0: return 'blue'
            else: return 'darkblue'
        return 'gray'


if __name__ == "__main__":
    print("Initializing Bangalore Grid...")
    grid = BangaloreGrid()
    
    # Test prediction adjustment
    test_prediction = {
        'temperature': 25.0,
        'precipitation': 0.5,
        'pressure': 1013.0
    }
    
    # Test locations covering more of the city
    locations = [
        (12.9716, 77.5946),  # City center
        (13.0098, 77.5053),  # Northwest
        (12.9300, 77.6850),  # Southeast
        (12.9850, 77.7480),  # East
        (12.9100, 77.5500),  # Southwest
        (13.0400, 77.6200),  # Northeast
    ]
    
    print("\nTesting microclimate adjustments:")
    predictions = {}
    for lat, lng in locations:
        adjusted = grid.adjust_prediction(test_prediction, (lat, lng))
        predictions[h3.latlng_to_cell(lat, lng, 9)] = adjusted
        print(f"\nLocation: {lat:.4f}, {lng:.4f}")
        print(f"Adjusted temp: {adjusted['temperature']:.2f}°C")
    
    print("\nCreating visualization...")
    map_viz = grid.create_visualization(predictions, 'temperature')
    map_viz.save('bangalore_weather.html')
    print("Visualization saved as 'bangalore_weather.html'")