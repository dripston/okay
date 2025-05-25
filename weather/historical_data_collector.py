import pandas as pd
import numpy as np
from datetime import datetime
from meteostat import Point, Daily
from pathlib import Path

class BangaloreHistoricalCollector:
    def __init__(self):
        # Bangalore coordinates and elevation
        self.latitude = 12.9716
        self.longitude = 77.5946
        self.elevation = 920  # meters
        self.data_dir = Path("c:/lastone/weather/data")
        self.data_dir.mkdir(exist_ok=True)
        
    def calculate_derived_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional weather parameters using available data"""
        
        # Diurnal Temperature Range (DTR)
        df['dtr'] = df['tmax'] - df['tmin']
        
        # Magnus-Tetens formula for saturation vapor pressure
        def calc_vapor_pressure(temp):
            return 6.112 * np.exp((17.67 * temp) / (temp + 243.5))
        
        # Calculate dewpoint and humidity
        df['tdew'] = df['tmin'] - ((100 - 90) / 5)  # Assuming 90% RH at Tmin
        df['rhum'] = 100 * (calc_vapor_pressure(df['tdew']) / calc_vapor_pressure(df['tavg']))
        
        # Wet-Bulb Temperature (simplified Stull's formula)
        df['twb'] = df['tavg'] * np.arctan(0.151977 * np.sqrt(df['rhum'] + 8.313659)) + \
                    np.arctan(df['tavg'] + df['rhum']) - \
                    np.arctan(df['rhum'] - 1.676331) + \
                    0.00391838 * (df['rhum'])**1.5 * np.arctan(0.023101 * df['rhum']) - 4.686035
        
        # Heat Index (when temp > 20°C)
        mask = df['tavg'] > 20
        df.loc[mask, 'hi'] = -8.78469475556 + \
            1.61139411 * df.loc[mask, 'tavg'] + \
            2.33854883889 * df.loc[mask, 'rhum'] - \
            0.14611605 * df.loc[mask, 'tavg'] * df.loc[mask, 'rhum'] - \
            0.012308094 * df.loc[mask, 'tavg']**2 - \
            0.0164248277778 * df.loc[mask, 'rhum']**2 + \
            0.002211732 * df.loc[mask, 'tavg']**2 * df.loc[mask, 'rhum'] + \
            0.00072546 * df.loc[mask, 'tavg'] * df.loc[mask, 'rhum']**2
        
        # Potential Evapotranspiration (Hargreaves equation)
        df['pet'] = 0.0023 * (df['tavg'] + 17.8) * df['dtr']**0.5 * 15.0
        
        # Monsoon Index (30-day rolling precipitation)
        df['monsoon_index'] = df['prcp'].rolling(window=30, min_periods=1).sum()
        
        return df
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean the weather data"""
        if df.empty:
            return df
            
        # Remove future dates
        current_date = datetime.now().date()
        df = df[df.index.date <= current_date]
        
        # Keep only relevant columns
        essential_columns = ['tavg', 'tmin', 'tmax', 'prcp']
        df = df[essential_columns]
        
        # Show initial missing values
        missing_pct = (df.isna().sum() / len(df)) * 100
        print("\nInitial missing values percentage:")
        for col in df.columns:
            print(f"{col}: {missing_pct[col]:.2f}%")
        
        # Fill missing temperatures using Mean Diurnal Variation (MDV)
        df_filled = df.copy()
        for column in ['tmin', 'tmax']:
            if df[column].isna().any():
                # Group by day of year and calculate mean
                day_means = df[column].groupby(df.index.dayofyear).mean()
                
                # Fill missing values with same day-of-year mean
                for idx in df[df[column].isna()].index:
                    day = idx.dayofyear
                    if not np.isnan(day_means[day]):
                        df_filled.loc[idx, column] = day_means[day]
        
        # Handle remaining gaps with linear interpolation
        df_filled = df_filled.interpolate(method='linear', limit=3)
        
        # Calculate tavg if missing using filled tmin/tmax
        df_filled['tavg'] = df_filled['tavg'].fillna(
            df_filled[['tmin', 'tmax']].mean(axis=1)
        )
        
        # Handle precipitation
        df_filled['prcp'] = df_filled['prcp'].fillna(0)
        
        # Show remaining missing values
        missing_after = (df_filled.isna().sum() / len(df_filled)) * 100
        print("\nRemaining missing values after filling:")
        for col in df_filled.columns:
            print(f"{col}: {missing_after[col]:.2f}%")
        
        # Final validation
        df_filled = df_filled[
            (df_filled['tavg'].between(10, 40)) &
            (df_filled['tmin'].between(5, 35)) &
            (df_filled['tmax'].between(15, 45))
        ]
        
        # Calculate derived weather parameters
        df_filled = self.calculate_derived_parameters(df_filled)
        
        print(f"\nRows after cleaning: {len(df_filled)}")
        return df_filled
    
    def fetch_historical_data(self, start_year: int = 2013):
        """Fetch Bangalore weather data using Meteostat"""
        try:
            # Use specific Bangalore station
            from meteostat import Stations
            
            # Find nearest station to our coordinates
            stations = Stations()
            stations = stations.nearby(self.latitude, self.longitude)
            station = stations.fetch(1)
            
            if station.empty:
                print("Error: No weather station found near Bangalore")
                return pd.DataFrame()
                
            # Set strict date range
            start = datetime(start_year, 1, 1)
            yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=1)
            
            print(f"Fetching data from {start.date()} to {yesterday.date()}...")
            
            # Fetch data using station ID
            station_id = station.index[0]
            data = Daily(station_id, start, yesterday)
            data.clear_cache()
            df = data.fetch()
            
            if df.empty:
                print("Warning: No weather data available for the selected station")
                return df
            
            # Strict date validation
            df = df[(df.index >= start) & (df.index <= yesterday)]
            df = df.sort_index()
            
            # Handle missing values before saving
            df['prcp'] = df['prcp'].fillna(0)  # No precipitation when missing
            df['tavg'] = df['tavg'].fillna(df[['tmin', 'tmax']].mean(axis=1))  # Average of min and max
            
            if len(df) > 0:
                print(f"Successfully retrieved {len(df)} daily records")
                print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
                df.to_csv(self.data_dir / "bangalore_historical.csv", 
                         date_format='%Y-%m-%d',
                         float_format='%.2f',
                         na_rep='NaN')  # Explicit NaN representation
            else:
                print("No valid data found for the specified date range")
            
            return df
                
        except Exception as e:
            print(f"Error fetching Meteostat data: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    collector = BangaloreHistoricalCollector()
    
    print("Fetching Bangalore historical weather data...")
    historical_data = collector.fetch_historical_data()
    
    print("Processing data...")
    processed_data = collector.process_data(historical_data)
    
    print("\nSample of historical data:")
    print(processed_data.head())
    
    print(f"\nTotal records: {len(processed_data)}")
    print(f"Date range: {processed_data.index.min()} to {processed_data.index.max()}")