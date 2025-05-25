import unittest
import numpy as np
import pandas as pd
from prediction_agent import WeatherPredictionAgent

class TestWeatherPredictionAgent(unittest.TestCase):
    def setUp(self):
        self.agent = WeatherPredictionAgent()
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self):
        dates = pd.date_range(start='2023-01-01', periods=60, freq='H')
        return pd.DataFrame({
            'temp_00': np.random.normal(25, 5, 60),
            'temp_03': np.random.normal(23, 5, 60),
            'wind_00': np.random.normal(10, 3, 60),
            'wind_03': np.random.normal(11, 3, 60),
            'pres': np.random.normal(1013, 5, 60)
        }, index=dates)
    
    def test_temporal_patterns(self):
        patterns = self.agent._extract_temporal_patterns(self.sample_data)
        self.assertIsInstance(patterns, dict)
        self.assertIn('daily', patterns)
        self.assertIn('seasonal', patterns)
        
    def test_spatial_grid(self):
        grid = self.agent._create_spatial_grid(self.sample_data)
        self.assertEqual(grid.shape[0], 7)  # 7 days
        self.assertEqual(grid.shape[1], 24)  # 24 hours