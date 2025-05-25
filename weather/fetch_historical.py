from meteostat import Point, Daily
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def fetch_bangalore_data():
    # Bangalore coordinates
    bangalore = Point(12.9716, 77.5946, 920)
    
    # Time period (10 years)
    start = datetime(2014, 1, 1)
    end = datetime(2023, 12, 31)
    
    # Fetch data
    print("Fetching Bangalore weather data from 2014-2023...")
    data = Daily(bangalore, start, end)
    df = data.fetch()
    
    # Basic data cleaning
    print("\nCleaning and processing data...")
    # Use recommended methods instead of fillna with 'method'
    df = df.ffill().bfill()
    
    # Remove columns with all missing values
    df = df.dropna(axis=1, how='all')
    
    # Keep only relevant columns
    essential_columns = ['tavg', 'tmin', 'tmax', 'prcp', 'wdir', 'wspd', 'pres']
    df = df[essential_columns]
    
    # Save data
    output_path = "c:/lastone/weather/data/bangalore_historical.csv"
    df.to_csv(output_path)
    print(f"\nData saved to {output_path}")
    
    # Print basic statistics
    print("\nData Overview:")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Total Days: {len(df)}")
    print("\nAvailable Variables:")
    for col in df.columns:
        missing = df[col].isna().sum()
        print(f"{col}: {missing} missing values")
        if missing == 0:
            print(f"    Range: {df[col].min():.1f} to {df[col].max():.1f}")
    
    # Plot temperature trends
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['tavg'], label='Average Temperature')
    plt.title('Bangalore Temperature Trend (2014-2023)')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig('c:/lastone/weather/data/temperature_trend.png')
    plt.close()

if __name__ == "__main__":
    fetch_bangalore_data()