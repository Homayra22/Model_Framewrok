import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Register the custom VAE class
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def vae_loss(inputs, reconstruction, z_mean, z_log_var):
    # Compute reconstruction loss
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, reconstruction))
    # Compute KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    return reconstruction_loss + kl_loss


class VAE(keras.Model):
    def __init__(self, encoder, decoder, num_columns, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.num_columns = num_columns

        # Metrics to track losses
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        # Forward pass through the encoder and decoder
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)

        # Add the VAE loss explicitly
        loss = vae_loss(inputs, reconstruction, z_mean, z_log_var)
        self.add_loss(loss)

        # Track metrics
        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(
            tf.reduce_mean(tf.keras.losses.mse(inputs, reconstruction))
        )
        self.kl_loss_tracker.update_state(
            -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        )

        return reconstruction
