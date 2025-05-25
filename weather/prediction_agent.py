from agents.temperature_agent import DeepTemperatureAgent, TransformerTempAgent
from agents.ensemble_agent import MetaEnsembleAgent, HierarchicalEnsemble
from agents.precipitation_agent import DeepPrecipitationAgent
from agents.wind_agent import WindPatternAgent
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import torch
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import entropy

class WeatherPredictionAgent:
    def __init__(self):
        self.temp_agents = {
            'deep': DeepTemperatureAgent(),
            'transformer': TransformerTempAgent()
        }
        self.precip_agent = DeepPrecipitationAgent()
        self.wind_agent = WindPatternAgent()
        self.meta_ensemble = HierarchicalEnsemble()
        self.temporal_scaler = MinMaxScaler()
        
    def _extract_temporal_patterns(self, data):
        daily_patterns = self._get_daily_cycles(data)
        weekly_patterns = self._get_weekly_patterns(data)
        seasonal_trends = self._get_seasonal_decomposition(data)
        fourier_features = self._extract_fourier_features(data)
        wavelet_features = self._extract_wavelet_features(data)
        
        return {
            'daily': daily_patterns,
            'weekly': weekly_patterns,
            'seasonal': seasonal_trends,
            'fourier': fourier_features,
            'wavelet': wavelet_features
        }
    
    def _get_daily_cycles(self, data):
        cycles = []
        for col in data.columns:
            if 'temp' in col or 'wind' in col:
                series = data[col].values
                decomposition = seasonal_decompose(series, period=24)
                cycles.append(decomposition.seasonal)
        return np.stack(cycles, axis=-1)
    
    def _extract_fourier_features(self, data):
        features = []
        for col in data.columns:
            fft = np.fft.fft(data[col].values)
            power = np.abs(fft)
            phase = np.angle(fft)
            features.extend([power, phase])
        return np.stack(features, axis=-1)
    
    def _extract_wavelet_features(self, data):
        import pywt
        features = []
        for col in data.columns:
            coeffs = pywt.wavedec(data[col].values, 'db4', level=4)
            features.extend(coeffs)
        return np.concatenate(features)
    
    def _calculate_entropy_features(self, data):
        entropy_features = []
        for col in data.columns:
            hist, _ = np.histogram(data[col].values, bins=50, density=True)
            entropy_features.append(entropy(hist))
        return np.array(entropy_features)
        self.spatial_features = None
        
    def prepare_spatiotemporal_data(self, data):
        # Create 3D tensors for ConvLSTM (samples, timesteps, features)
        temporal_features = self._extract_temporal_patterns(data)
        spatial_features = self._create_spatial_grid(data)
        
        return {
            'temporal': temporal_features,
            'spatial': spatial_features,
            'combined': np.concatenate([temporal_features, spatial_features], axis=-1)
        }
    
    def _extract_temporal_patterns(self, data):
        # Extract complex temporal patterns
        daily_patterns = self._get_daily_cycles(data)
        weekly_patterns = self._get_weekly_patterns(data)
        seasonal_trends = self._get_seasonal_decomposition(data)
        
        return np.stack([daily_patterns, weekly_patterns, seasonal_trends], axis=-1)
    
    def _create_spatial_grid(self, data):
        # Create spatial representation for local patterns
        grid_size = (7, 24)  # 7 days x 24 hours
        spatial_grid = np.zeros((*grid_size, len(data.columns)))
        
        for i, col in enumerate(data.columns):
            values = data[col].values
            spatial_grid[:, :, i] = values.reshape(-1, 24)[:7]
        
        return spatial_grid
