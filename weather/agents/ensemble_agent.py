from sklearn.ensemble import StackingRegressor, VotingRegressor
import optuna
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
# Remove this line since we'll use tf.lite directly
# import tensorflow.lite as tflite
from catboost import CatBoostRegressor

# Add custom neural network classes
from models.bayesian_nn import BayesianNeuralNetwork
from models.physics_nn import PhysicsGuidedNN, TemperatureConstraint, PressureConstraint, HumidityConstraint
from models.transformers import TemporalFusionTransformer, SpatialTransformer

class MetaEnsembleAgent:
    def __init__(self):
        self.study = optuna.create_study(direction="minimize")
        self.best_weights = None
        self.base_models = self._initialize_models()
        self.meta_learner = self._build_meta_learner()
        self.gp_models = {}
        self.uncertainty_estimator = self._build_uncertainty_estimator()
        self.model_memory = {}  # Store historical performance
        
    def _build_meta_learner(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.Dense(64, activation='selu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32)
        ])
        # Use tf.lite instead of tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return tflite_model

    def _build_temporal_ensemble(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),   # Reduced from 256
            tf.keras.layers.LSTM(32),                         # Reduced from 128
            tf.keras.layers.Dense(32, activation='selu'),
            tf.keras.layers.Dense(16)
        ])
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()
        
    def _build_uncertainty_estimator(self):
        return BayesianNeuralNetwork(
            hidden_layers=[256, 128],
            num_monte_carlo=1000
        )
    
    def _initialize_models(self):
        models = super()._initialize_models()
        models.update({
            'physics_guided': self._build_physics_guided_nn(),
            'temporal_fusion': self._build_temporal_fusion(),
            'spatial_attention': self._build_spatial_attention()
        })
        return models
    
    def _build_physics_guided_nn(self):
        return PhysicsGuidedNN(
            physical_constraints=[
                TemperatureConstraint(),
                PressureConstraint(),
                HumidityConstraint()
            ]
        )
    
    def _build_temporal_fusion(self):
        return TemporalFusionTransformer(
            num_layers=4,
            d_model=256,
            num_heads=8,
            dropout=0.2
        )
    
    def _build_spatial_attention(self):
        return SpatialTransformer(
            patch_size=4,
            num_patches=64,
            num_layers=6
        )
        
    def train(self, predictions_dict, actual_values):
        X_train = self._prepare_ensemble_features(predictions_dict)
        
        # Train base models
        for model_name, model in self.base_models.items():
            model.fit(X_train, actual_values)
            
        # Optimize ensemble weights
        self.optimize_weights(predictions_dict, actual_values)
        
        # Train Gaussian Process for uncertainty estimation
        self._train_uncertainty_models(X_train, actual_values)
    
    def _train_uncertainty_models(self, X, y):
        for feature in X.columns:
            kernel = ConstantKernel() * RBF() + Matern(nu=2.5)
            self.gp_models[feature] = GaussianProcessRegressor(
                kernel=kernel,
                alpha=0.1,
                normalize_y=True,
                n_restarts_optimizer=15
            ).fit(X[[feature]], y)
    
    def predict(self, features):
        base_predictions = {}
        uncertainties = {}
        
        # Get predictions from base models
        for name, model in self.base_models.items():
            base_predictions[name] = model.predict(features)
            
        # Get uncertainty estimates
        for feature in features.columns:
            pred, std = self.gp_models[feature].predict(features[[feature]], return_std=True)
            uncertainties[feature] = std
            
        # Combine predictions using optimized weights
        weighted_pred = self._combine_predictions(base_predictions, self.best_weights)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(uncertainties)
        
        return {
            'prediction': weighted_pred,
            'confidence': confidence,
            'uncertainties': uncertainties
        }
    
    def optimize_weights(self, predictions, actual):
        def objective(trial):
            weights = {
                'temp': trial.suggest_float('temp_weight', 0.2, 0.6),
                'precip': trial.suggest_float('precip_weight', 0.2, 0.4),
                'wind': trial.suggest_float('wind_weight', 0.2, 0.4)
            }
            
            # Add regularization to prevent overfitting
            reg_term = trial.suggest_float('regularization', 0.01, 0.1)
            error = self._calculate_weighted_error(predictions, actual, weights)
            
            return error + reg_term * sum(w**2 for w in weights.values())
            
        # Run optimization
        self.study.optimize(objective, n_trials=100)
        self.best_weights = self.study.best_params
    
    def _calculate_confidence(self, uncertainties):
        # Convert uncertainties to confidence scores
        confidence_scores = [1 / (1 + np.exp(u.mean())) for u in uncertainties.values()]
        return np.mean(confidence_scores)
    
    def _calculate_weighted_error(self, predictions, actual, weights):
        weighted_pred = sum(w * p for w, p in zip(weights.values(), predictions.values()))
        return mean_squared_error(actual, weighted_pred, squared=False)


class HierarchicalEnsemble:
    def __init__(self):
        self.level1_models = self._init_level1()
        self.level2_models = self._init_level2()
        self.meta_learner = self._init_meta_learner()
        
    def _init_level1(self):
        return {
            'temporal': self._build_temporal_ensemble(),
            'spatial': self._build_spatial_ensemble(),
            'combined': self._build_combined_ensemble()
        }
    
    def _build_temporal_ensemble(self):
        return tf.keras.Sequential([
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(64, activation='selu'),
            tf.keras.layers.Dense(32)
        ])
    
    def _build_spatial_ensemble(self):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32)
        ])