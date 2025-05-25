import tensorflow as tf
import numpy as np
from models.bayesian_nn import BayesianNeuralNetwork
from models.physics_nn import PhysicsGuidedNN, TemperatureConstraint, PressureConstraint, HumidityConstraint
from models.transformers import TemporalFusionTransformer, SpatialTransformer

def test_bayesian_nn():
    print("Testing Bayesian Neural Network...")
    model = BayesianNeuralNetwork(hidden_layers=[64, 32])
    test_input = tf.random.normal([10, 32])
    try:
        # Call the model directly as a callable
        output = model(test_input)  # Keras models are callable by default
        print("BayesianNN Test Passed!")
    except Exception as e:
        print(f"BayesianNN Test Failed: {e}")

def test_physics_nn():
    print("\nTesting Physics Guided Neural Network...")
    constraints = [TemperatureConstraint(), PressureConstraint(), HumidityConstraint()]
    model = PhysicsGuidedNN(physical_constraints=constraints)
    test_input = tf.random.normal([10, 32])
    try:
        output = model.model(test_input)
        print("PhysicsNN Test Passed!")
    except Exception as e:
        print(f"PhysicsNN Test Failed: {e}")

def test_transformers():
    print("\nTesting Transformers...")
    temporal_transformer = TemporalFusionTransformer(
        num_layers=4, d_model=256, num_heads=8, dropout=0.1
    )
    spatial_transformer = SpatialTransformer(
        patch_size=4, num_patches=64, num_layers=6
    )
    
    test_input = tf.random.normal([10, 32, 256])
    try:
        # Call transformers directly as callables
        temp_output = temporal_transformer(test_input)  # Direct call
        spat_input = tf.reshape(test_input, [10, -1, 256])
        spat_output = spatial_transformer(spat_input)  # Direct call
        print("Transformers Test Passed!")
    except Exception as e:
        print(f"Transformers Test Failed: {e}")

if __name__ == "__main__":
    test_bayesian_nn()
    test_physics_nn()
    test_transformers()