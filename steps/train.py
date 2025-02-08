from zenml import step
from tensorflow import keras
from models.vae import VAE, build_encoder, build_decoder


@step
def train_vae(data: dict) -> VAE:
    """
    Training step for the VAE model.

    Args:
        data (dict): Dictionary containing training and validation data.
            Expected keys: "x_train", "y_train", "x_val", "y_val".

    Returns:
        VAE: Trained VAE model.
    """
    # Build the encoder and decoder
    encoder = build_encoder()
    decoder = build_decoder()

    # Instantiate the VAE model
    vae = VAE(encoder, decoder, num_columns=6)

    # Compile the model
    vae.compile(optimizer=keras.optimizers.Adam())

    # Train the model
    vae.fit(
        data["x_train"],  # Training input data
        data["y_train"],  # Training target data
        validation_data=(data["x_val"], data["y_val"]),  # Validation data
        epochs=100,  # Number of epochs
        batch_size=1000,  # Batch size
    )

    return vae
