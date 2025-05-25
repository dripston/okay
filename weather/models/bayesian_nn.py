import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np  # Add numpy import

class BayesianDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(BayesianDense, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_shape[-1], self.units],
            initializer='glorot_normal',
            trainable=True,
            dtype=self.dtype
        )
        self.bias = self.add_weight(
            'bias',
            shape=[self.units],
            initializer='zeros',
            trainable=True,
            dtype=self.dtype
        )
        self.built = True

    def call(self, inputs, training=None):
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

class BayesianNeuralNetwork(tf.keras.Model):
    def __init__(self, hidden_layers=[64, 32]):
        super().__init__()
        # Initialize Bayesian layers
        self.dense_layers = []
        for units in hidden_layers:
            self.dense_layers.append(BayesianDense(units, activation='relu'))
        self.output_dense = BayesianDense(4)  # Output 4 features: temp, humidity, pressure, wind
        
        # Compile the model
        self.compile(
            optimizer='adam',
            loss='mse',  # Mean squared error for regression
            metrics=['mae']  # Mean absolute error for monitoring
        )

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x, training=training)
        return self.output_dense(x, training=training)
    
    def predict_with_uncertainty(self, inputs, num_samples=100):
        """
        Make predictions with uncertainty estimates
        
        Args:
            inputs: Input data
            num_samples: Number of forward passes to estimate uncertainty
            
        Returns:
            mean_prediction: Mean prediction values
            std_prediction: Standard deviation (uncertainty)
        """
        predictions = []
        
        # Multiple forward passes
        for _ in range(num_samples):
            pred = self(inputs, training=True).numpy()
            predictions.append(pred)
            
        # Stack predictions and compute statistics
        predictions = np.stack(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        return mean_prediction, std_prediction