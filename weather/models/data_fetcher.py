import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import os
import pickle
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class WeatherDataFetcher:
    def __init__(self):
        """Initialize the WeatherDataFetcher with city coordinates, base values, and scaling parameters."""
        # Add cache configuration first
        self.cache_dir = Path('d:/lastone/weather/cache')
        self.cache_duration_days = 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.city_coords = {
            'Whitefield': (12.9698, 77.7500),
            'Electronic_City': (12.8458, 77.6631),  # Fixed coordinate
            'Koramangala': (12.9279, 77.6271),
            'Indiranagar': (12.9719, 77.6412),
            'Marathahalli': (12.9591, 77.6960),
            'Hebbal': (13.0355, 77.5997),
            'Bangalore_Central': (12.9716, 77.5946)
        }
        # Add city-specific base values
        self.city_base_values = {
            'Whitefield': {'temperature': 28.5, 'humidity': 65, 'pressure': 913, 'wind_speed': 8},
            'Electronic_City': {'temperature': 28.0, 'humidity': 63, 'pressure': 912, 'wind_speed': 9},
            'Koramangala': {'temperature': 28.2, 'humidity': 64, 'pressure': 913, 'wind_speed': 7},
            'Indiranagar': {'temperature': 28.3, 'humidity': 65, 'pressure': 913, 'wind_speed': 8},
            'Marathahalli': {'temperature': 28.4, 'humidity': 66, 'pressure': 913, 'wind_speed': 9},
            'Hebbal': {'temperature': 27.8, 'humidity': 67, 'pressure': 912, 'wind_speed': 8},
            'Bangalore_Central': {'temperature': 28.0, 'humidity': 65, 'pressure': 913, 'wind_speed': 7}
        }
        self.sequence_length = 24
        self.ranges = {
            'temperature': (20, 35),
            'humidity': (40, 90),
            'pressure': (900, 1015),
            'wind_speed': (0, 20)
        }
        
        # Add scaling parameters based on typical Bangalore weather ranges
        self.feature_means = {
            'temperature': 27.5,  # °C
            'humidity': 65.0,     # %
            'pressure': 957.5,    # hPa
            'wind_speed': 10.0    # km/h
        }
        
        self.feature_stds = {
            'temperature': 7.5,   # °C
            'humidity': 25.0,     # %
            'pressure': 57.5,     # hPa
            'wind_speed': 10.0    # km/h
        }

    def scale_features(self, data):
        """Scale features using standardization (z-score)"""
        if isinstance(data, list):
            data = np.array(data)
            
        scaled_data = np.zeros_like(data, dtype=np.float32)
        
        for i, feature in enumerate(['temperature', 'humidity', 'pressure', 'wind_speed']):
            scaled_data[..., i] = (data[..., i] - self.feature_means[feature]) / self.feature_stds[feature]
            
        return scaled_data

    def unscale_features(self, scaled_data):
        """Reverse the scaling transformation"""
        if isinstance(scaled_data, list):
            scaled_data = np.array(scaled_data)
            
        data = np.zeros_like(scaled_data, dtype=np.float32)
        
        for i, feature in enumerate(['temperature', 'humidity', 'pressure', 'wind_speed']):
            data[..., i] = (scaled_data[..., i] * self.feature_stds[feature]) + self.feature_means[feature]
            
        return data

    def _get_raw_data_from_api(self, latitude, longitude, start_date, end_date):
        """Get weather data from local historical CSV instead of API"""
        try:
            # Read from local CSV file
            csv_path = Path('d:/lastone/weather/data/bangalore_historical.csv')
            if not csv_path.exists():
                raise FileNotFoundError("Historical weather data file not found")
                
            df = pd.read_csv(csv_path)
            df['time'] = pd.to_datetime(df['time'])
            
            # Filter data for the requested date range
            mask = (df['time'].dt.date >= pd.to_datetime(start_date).date()) & \
                   (df['time'].dt.date <= pd.to_datetime(end_date).date())
            df = df[mask].copy()
            
            if len(df) == 0:
                raise ValueError(f"No data available for the date range {start_date} to {end_date}")
            
            # Create hourly data points (since CSV is daily)
            hourly_data = []
            for _, row in df.iterrows():
                for hour in range(24):
                    # Add daily variation
                    temp_var = 2.0 * np.sin((hour - 14) * np.pi / 12)  # Peak at 2 PM
                    humid_var = 10.0 * np.sin((hour - 4) * np.pi / 12)  # Peak at 4 AM
                    wind_var = 2.0 * np.sin((hour - 13) * np.pi / 12)   # Peak at 1 PM
                    
                    hourly_data.append({
                        'temperature': row['tavg'] + temp_var,
                        'humidity': 65.0 + humid_var,  # Base humidity of 65%
                        'pressure': row['pres'],
                        'wind_speed': row['wspd'] + wind_var
                    })
            
            return pd.DataFrame(hourly_data)
            
        except Exception as e:
            logging.error(f"Error reading historical data: {str(e)}")
            raise

    def _clean_weather_data(self, df):
        """Clean and validate weather data"""
        try:
            for column in df.columns:
                # Replace invalid values with NaN
                df[column] = pd.to_numeric(df[column], errors='coerce')
                
                # Interpolate missing values
                df[column] = df[column].interpolate(method='linear')
                
                # Fill remaining NaN with column mean
                df[column] = df[column].fillna(df[column].mean())
                
                # Clip to valid ranges
                min_val, max_val = self.ranges[column]
                df[column] = df[column].clip(min_val, max_val)
            
            return df
            
        except Exception as e:
            logging.error(f"Data cleaning failed: {str(e)}")
            raise

    def _create_sequences(self, df):
        """Create sequences and targets from cleaned data"""
        try:
            sequences = []
            targets = []
            
            for i in range(len(df) - self.sequence_length):
                sequence = df.iloc[i:i+self.sequence_length].values
                target = df.iloc[i+self.sequence_length].values
                sequences.append(sequence)
                targets.append(target)
                
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            logging.error(f"Sequence creation failed: {str(e)}")
            raise

    def fetch_historical_data(self, latitude, longitude, start_date, end_date):
        """Main method to fetch and process historical weather data"""
        try:
            # Get raw data from API
            df = self._get_raw_data_from_api(latitude, longitude, start_date, end_date)
            
            # Clean and validate data
            df = self._clean_weather_data(df)
            
            # Create sequences
            sequences, targets = self._create_sequences(df)
            
            # Scale the sequences and targets
            scaled_sequences = self.scale_features(sequences)
            scaled_targets = self.scale_features(targets)
            
            return scaled_sequences, scaled_targets
            
        except Exception as e:
            logging.error(f"Historical data processing failed: {str(e)}")
            raise

    def get_current_sequence(self, city_name):
        if city_name not in self.city_coords:
            raise ValueError(f"Unknown city: {city_name}. Available cities: {list(self.city_coords.keys())}")
            
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        sequences, _ = self.fetch_historical_data(
            latitude=self.city_coords[city_name][0],
            longitude=self.city_coords[city_name][1],
            start_date=start_date,
            end_date=end_date
        )
        
        if len(sequences) == 0:
            # Create a synthetic 24-hour sequence using base values
            base_values = self.city_base_values[city_name]
            hour = datetime.now().hour
            
            sequence = []
            for i in range(self.sequence_length):
                adj_hour = (hour - (self.sequence_length - 1 - i)) % 24
                
                # Add daily variations
                temp_var = 2.0 * np.sin((adj_hour - 14) * np.pi / 12)  # Peak at 2 PM
                humid_var = -5.0 * np.sin((adj_hour - 4) * np.pi / 12)  # Peak at 4 AM
                wind_var = 2.0 * np.sin((adj_hour - 13) * np.pi / 12)   # Peak at 1 PM
                
                sequence.append([
                    base_values['temperature'] + temp_var,
                    base_values['humidity'] + humid_var,
                    base_values['pressure'],
                    base_values['wind_speed'] + wind_var
                ])
            
            return np.array(sequence)
            
        # Return the last 24 hours of data
        return sequences[-1]

    def _get_cache_path(self, latitude, longitude, start_date, end_date, days):
        """Generate a unique cache file path for the given parameters"""
        cache_key = f"weather_{latitude}_{longitude}_{start_date}_{end_date}_{days}"
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path):
        """Check if cache file exists and is recent enough"""
        if not cache_path.exists():
            return False
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age.days < self.cache_duration_days

    def get_historical_data(self, latitude, longitude, start_date, end_date):
        """Get historical weather data for a location"""
        try:
            # Check cache first
            cache_key = f"{latitude}_{longitude}_{start_date}_{end_date}"
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # If cache exists and is recent, use it
            if cache_file.exists() and self._is_cache_valid(cache_file):
                logging.info(f"Using cached data for {cache_key}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # If no cache, try to get data from API
            logging.info(f"Fetching historical data for coordinates ({latitude}, {longitude})")
            
            # Try to use local CSV file first
            try:
                csv_path = Path('d:/lastone/weather/data/bangalore_historical.csv')
                if csv_path.exists():
                    # Process CSV data
                    df = pd.read_csv(csv_path)
                    df['time'] = pd.to_datetime(df['time'])
                    
                    # Create hourly data structure
                    hourly_data = {
                        'time': [],
                        'temperature_2m': [],
                        'relative_humidity_2m': [],
                        'pressure_msl': [],
                        'wind_speed_10m': [],
                        'precipitation': []
                    }
                    
                    # Generate hourly data from daily data
                    for _, row in df.iterrows():
                        base_date = pd.to_datetime(row['time'])
                        for hour in range(24):
                            current_time = base_date + pd.Timedelta(hours=hour)
                            
                            # Skip if outside requested range
                            if current_time.strftime("%Y-%m-%d") < start_date or current_time.strftime("%Y-%m-%d") > end_date:
                                continue
                                
                            # Add daily variation
                            temp_var = 2.0 * np.sin((hour - 14) * np.pi / 12)  # Peak at 2 PM
                            humid_var = 10.0 * np.sin((hour - 4) * np.pi / 12)  # Peak at 4 AM
                            wind_var = 2.0 * np.sin((hour - 13) * np.pi / 12)   # Peak at 1 PM
                            
                            hourly_data['time'].append(current_time.strftime("%Y-%m-%dT%H:%M"))
                            hourly_data['temperature_2m'].append(row.get('tavg', 25.0) + temp_var)
                            hourly_data['relative_humidity_2m'].append(65.0 + humid_var)
                            hourly_data['pressure_msl'].append(row.get('pres', 1013.0))
                            hourly_data['wind_speed_10m'].append(row.get('wspd', 8.0) + wind_var)
                            hourly_data['precipitation'].append(row.get('prcp', 0.0))
                    
                    # Create result structure
                    result = {'hourly': hourly_data}
                    
                    # Cache the result
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    return result
                    
            except Exception as e:
                logging.warning(f"Could not use local CSV file: {str(e)}")
                # Fall back to synthetic data
                
            # Generate synthetic data if CSV fails
            logging.info("Generating synthetic weather data")
            result = self._generate_synthetic_data(latitude, longitude, start_date, end_date)
            
            # Cache the result
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
            
        except Exception as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            # Return synthetic data as fallback
            return self._generate_synthetic_data(latitude, longitude, start_date, end_date)

    def _generate_synthetic_data(self, latitude, longitude, start_date, end_date):
        """Generate synthetic weather data when API fails"""
        logging.info("Generating synthetic weather data")
        
        # Convert dates to datetime objects
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Calculate number of hours
        hours = int((end_dt - start_dt).total_seconds() / 3600) + 1
        
        # Get base values for the nearest city
        nearest_city = 'Bangalore_Central'  # Default
        min_dist = float('inf')
        for city, (city_lat, city_lon) in self.city_coords.items():
            dist = ((latitude - city_lat)**2 + (longitude - city_lon)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_city = city
        
        base_values = self.city_base_values.get(nearest_city, {
            'temperature': 28.0,
            'humidity': 65.0,
            'pressure': 913.0,
            'wind_speed': 8.0
        })
        
        # Generate hourly data
        hourly_data = {
            'time': [],
            'temperature_2m': [],
            'relative_humidity_2m': [],
            'pressure_msl': [],
            'wind_speed_10m': [],
            'precipitation': []
        }
        
        for hour in range(hours):
            current_time = start_dt + pd.Timedelta(hours=hour)
            day_of_year = current_time.dayofyear
            hour_of_day = current_time.hour
            
            # Add seasonal and daily variations
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
            daily_factor = np.sin(2 * np.pi * (hour_of_day - 14) / 24)  # Peak at 2 PM
            
            # Temperature: seasonal ±3°C, daily ±5°C
            temp = base_values['temperature'] + 3 * seasonal_factor + 5 * daily_factor
            
            # Humidity: inverse to temperature
            humidity = base_values['humidity'] - 15 * daily_factor
            
            # Pressure: small random variations
            pressure = base_values['pressure'] + np.random.normal(0, 1)
            
            # Wind: more in afternoon
            wind = base_values['wind_speed'] + 3 * daily_factor + np.random.normal(0, 1)
            
            # Precipitation: higher chance when humidity is high
            precip = 0.0
            if humidity > 80 and np.random.random() < 0.3:
                precip = np.random.exponential(1.0)
            
            # Add to hourly data
            hourly_data['time'].append(current_time.strftime("%Y-%m-%dT%H:%M"))
            hourly_data['temperature_2m'].append(float(temp))
            hourly_data['relative_humidity_2m'].append(float(max(0, min(100, humidity))))
            hourly_data['pressure_msl'].append(float(pressure))
            hourly_data['wind_speed_10m'].append(float(max(0, wind)))
            hourly_data['precipitation'].append(float(precip))
        
        return {'hourly': hourly_data}

    def get_current_sequence(self, city_name):
        if city_name not in self.city_coords:
            raise ValueError(f"Unknown city: {city_name}. Available cities: {list(self.city_coords.keys())}")
            
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        sequences, _ = self.fetch_historical_data(
            latitude=self.city_coords[city_name][0],
            longitude=self.city_coords[city_name][1],
            start_date=start_date,
            end_date=end_date
        )
        
        if len(sequences) == 0:
            # Create a synthetic 24-hour sequence using base values
            base_values = self.city_base_values[city_name]
            hour = datetime.now().hour
            
            sequence = []
            for i in range(self.sequence_length):
                adj_hour = (hour - (self.sequence_length - 1 - i)) % 24
                
                # Add daily variations
                temp_var = 2.0 * np.sin((adj_hour - 14) * np.pi / 12)  # Peak at 2 PM
                humid_var = -5.0 * np.sin((adj_hour - 4) * np.pi / 12)  # Peak at 4 AM
                wind_var = 2.0 * np.sin((adj_hour - 13) * np.pi / 12)   # Peak at 1 PM
                
                sequence.append([
                    base_values['temperature'] + temp_var,
                    base_values['humidity'] + humid_var,
                    base_values['pressure'],
                    base_values['wind_speed'] + wind_var
                ])
            
            return np.array(sequence)
            
        # Return the last 24 hours of data
        return sequences[-1]