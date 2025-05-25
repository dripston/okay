import tensorflow as tf

class TemporalFusionTransformer(tf.keras.Model):
    def __init__(self, num_layers=4, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model * 4, activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model)
        ])
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        
        # Add prediction head
        # Change prediction layer to output 4 features
        self.prediction_layer = tf.keras.layers.Dense(4)  # Changed from 1 to 4
        
    def call(self, inputs, training=None, mask=None):
        # Normalization and Attention
        x1 = self.norm1(inputs)
        attn_output = self.mha(
            query=x1,
            key=x1,
            value=x1,
            attention_mask=mask,  # Changed from mask to attention_mask
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = inputs + attn_output
        
        # Feed-forward network
        x2 = self.norm2(out1)
        ffn_output = self.ffn(x2, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # Generate predictions
        # Add global average pooling before prediction
        x = tf.reduce_mean(ffn_output, axis=1)  # Average across time dimension
        predictions = self.prediction_layer(x)   # Generate 4 predictions
        return predictions

class SpatialTransformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=4
        )
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(4)  # Already correct

    def call(self, inputs, training=None):
        # Add positional encoding
        attention_output = self.attention(inputs, inputs)
        x = self.layer_norm1(inputs + attention_output)
        
        # Dense layers
        x = self.dense1(x)
        x = self.layer_norm2(x)
        x = self.dense2(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)  # Already correct
        return self.output_layer(x)