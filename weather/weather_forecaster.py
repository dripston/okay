from historical_data_collector import BangaloreHistoricalCollector
from prediction_agent import WeatherPredictionAgent
from pathlib import Path

def main():
    # Initialize collectors and agents
    collector = BangaloreHistoricalCollector()
    predictor = WeatherPredictionAgent()
    
    # Fetch and process historical data
    print("Fetching historical weather data...")
    historical_data = collector.fetch_historical_data()
    processed_data = collector.process_data(historical_data)
    
    if processed_data.empty:
        print("Error: No historical data available for training")
        return
    
    # Train prediction models
    print("\nTraining prediction models...")
    predictor.train(processed_data)
    
    # Make predictions for different windows
    print("\nGenerating forecasts...")
    recent_data = processed_data.iloc[-30:]  # Use last 30 days for prediction
    
    for window in predictor.forecast_windows:
        print(f"\n{window}-day forecast:")
        forecast = predictor.predict(recent_data, window)
        print(forecast)
        
        # Save forecasts
        forecast.to_csv(
            predictor.data_dir / f'bangalore_forecast_{window}days.csv',
            date_format='%Y-%m-%d',
            float_format='%.2f'
        )

if __name__ == "__main__":
    main()