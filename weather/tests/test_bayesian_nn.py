import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
import numpy as np
import unittest

from models.bayesian_nn import BayesianDense, BayesianNeuralNetwork

class TestBayesianNN(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Create test data
        self.test_input = np.random.normal(size=(10, 4))
        self.test_output = np.random.normal(size=(10, 4))
        
        # Create model instance
        self.model = BayesianNeuralNetwork(hidden_layers=[32, 16])
    
    def test_bayesian_dense_layer(self):
        """Test BayesianDense layer initialization and forward pass"""
        # Create a layer
        layer = BayesianDense(units=10, activation='relu')
        
        # Test forward pass
        inputs = tf.random.normal((5, 4))
        outputs = layer(inputs)
        
        # Check output shape
        self.assertEqual(outputs.shape, (5, 10))
        
        # Check if weights are created
        self.assertTrue(hasattr(layer, 'kernel'))
        self.assertTrue(hasattr(layer, 'bias'))
        
        # Check weights shapes
        self.assertEqual(layer.kernel.shape, (4, 10))
        self.assertEqual(layer.bias.shape, (10,))
    
    def test_bayesian_nn_initialization(self):
        """Test BayesianNeuralNetwork initialization"""
        # Check if layers are created
        self.assertEqual(len(self.model.dense_layers), 2)
        self.assertEqual(self.model.dense_layers[0].units, 32)
        self.assertEqual(self.model.dense_layers[1].units, 16)
        self.assertEqual(self.model.output_dense.units, 4)
        
        # Check if model is compiled
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)
    
    def test_bayesian_nn_forward_pass(self):
        """Test BayesianNeuralNetwork forward pass"""
        # Test forward pass
        outputs = self.model(self.test_input)
        
        # Check output shape
        self.assertEqual(outputs.shape, (10, 4))
        
        # Check if outputs are finite
        self.assertTrue(np.all(np.isfinite(outputs)))
    
    def test_bayesian_nn_training(self):
        """Test BayesianNeuralNetwork training"""
        # Train for a few epochs
        history = self.model.fit(
            self.test_input, 
            self.test_output,
            epochs=5,
            batch_size=2,
            verbose=0
        )
        
        # Check if loss decreases
        losses = history.history['loss']
        self.assertGreaterEqual(losses[0], losses[-1])
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation functionality"""
        # Import numpy for the test
        import numpy as np
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self.model.predict_with_uncertainty(
            self.test_input, 
            num_samples=10
        )
        
        # Check shapes
        self.assertEqual(mean_pred.shape, (10, 4))
        self.assertEqual(std_pred.shape, (10, 4))
        
        # Check if uncertainty values are positive
        self.assertTrue(np.all(std_pred >= 0))
        
        # Check if mean predictions are finite
        self.assertTrue(np.all(np.isfinite(mean_pred)))
    
    def test_different_inputs(self):
        """Test model with different input shapes"""
        # Test with single sample
        single_input = np.random.normal(size=(1, 4))
        single_output = self.model(single_input)
        self.assertEqual(single_output.shape, (1, 4))
        
        # Test with larger batch
        large_input = np.random.normal(size=(100, 4))
        large_output = self.model(large_input)
        self.assertEqual(large_output.shape, (100, 4))
    
    def test_save_and_load(self):
        """Test model saving and loading"""
        # First call the model to create variables
        _ = self.model(self.test_input)
        
        # Save model weights
        self.model.save_weights('d:/lastone/weather/tests/temp_bayesian_model.h5')
        
        # Create new model
        new_model = BayesianNeuralNetwork(hidden_layers=[32, 16])
        
        # Call the new model to create variables
        _ = new_model(self.test_input)
        
        # Load weights
        new_model.load_weights('d:/lastone/weather/tests/temp_bayesian_model.h5')
        
        # Compare predictions
        original_pred = self.model(self.test_input).numpy()
        loaded_pred = new_model(self.test_input).numpy()
        
        # Check if predictions are the same
        np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-5)
        
        # Clean up
        import os
        if os.path.exists('d:/lastone/weather/tests/temp_bayesian_model.h5'):
            os.remove('d:/lastone/weather/tests/temp_bayesian_model.h5')

if __name__ == '__main__':
    # Fix for TensorFlow GPU memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    unittest.main()