# Add Flask imports and create a blueprint at the top of the file
from flask import Blueprint, jsonify, render_template
import csv  # Add this import if it's not already present
import os  # Add missing import
from datetime import datetime  # Add missing import

# Create blueprint
long_term_bp = Blueprint('long_term', __name__, url_prefix='/long-term')

# Add these routes after the existing LongTermPredictionSystem class
@long_term_bp.route('/forecast')
def long_term_forecast():
    """Render the long-term forecast page."""
    return render_template('long_term_forecast.html')

@long_term_bp.route('/api/forecast')
def get_long_term_forecast():
    """API endpoint to get long-term forecast data."""
    try:
        # Check if CSV file exists
        csv_file = "d:\\lastone\\weather\\output\\six_month_forecast.csv"
        if not os.path.exists(csv_file):
            # Generate forecast data if it doesn't exist
            long_term_system = LongTermPredictionSystem()
            result = long_term_system.predict_six_months()
            
            if 'error' in result:
                return jsonify({
                    'error': True,
                    'message': result['error']
                }), 500
        
        # Read data from CSV
        forecast_data = []
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert string values to appropriate types
                forecast_data.append({
                    'date': row['date'],
                    'condition': determine_condition(float(row.get('prcp', 0)), float(row.get('tavg', 25))),
                    'tavg': float(row.get('tavg', 0)),
                    'tmin': float(row.get('tmin', 0)),
                    'tmax': float(row.get('tmax', 0)),
                    'humidity': float(row.get('humidity', 70)) if 'humidity' in row else 70,
                    'wspd': float(row.get('wspd', 0)),
                    'pres': float(row.get('pres', 0)),
                    'prcp': float(row.get('prcp', 0))
                })
        
        return jsonify({
            'city': 'Bangalore',
            'forecast': forecast_data,
            'is_real_data': True,
            'last_updated': datetime.now().isoformat(),
            'source': 'CSV data'
        })
    
    except Exception as e:
        return jsonify({
            'error': True,
            'message': str(e)
        }), 500

def determine_condition(precipitation, temperature):
    """Determine weather condition based on precipitation and temperature."""
    if precipitation > 10:
        return 'Heavy Rain'
    elif precipitation > 5:
        return 'Rain'
    elif precipitation > 0:
        return 'Light Rain'
    elif temperature > 30:
        return 'Sunny'
    elif temperature > 25:
        return 'Partly Cloudy'
    else:
        return 'Cloudy'

        