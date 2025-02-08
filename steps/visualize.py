import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from zenml import step


@step
def visualize_results(vae: VAE, data: dict):
    """
    Visualize the VAE predictions on the validation dataset.

    Args:
        vae (VAE): Trained VAE model.
        data (dict): Dictionary containing validation data.
            Expected keys: "x_1201", "scaler_Y".
    """
    # Predict on 1201 data
    _, _, y_pred_1201 = vae.encoder.predict(data["x_1201"])
    y_pred_1201 = vae.decoder.predict(y_pred_1201)
    y_pred_1201 = data["scaler_Y"].inverse_transform(y_pred_1201)

    # Reshape predictions into a 2D grid
    prediction_on_1201_2D = y_pred_1201.reshape(1201, 1201)
    data_set1 = np.transpose(prediction_on_1201_2D)
    data_set1 = np.flipud(data_set1)

    # Define the colormap
    colors = [
        (0.0, 0.0, 0.0),  # Black
        (0.0, 0.0, 0.3),  # Darker blue
        (0.0, 0.0, 0.5),  # Dark blue
        (0.0, 0.0, 0.7),  # Blue
        (0.0, 0.0, 1.0),  # Blue
        (0.0, 0.4, 0.8),  # Lighter blue
        (0.0, 0.6, 0.2),  # Dark green
        (0.0, 0.7, 0.3),  # Green
        (0.2, 0.8, 0.4),  # Light green
        (1.0, 0.8, 0.0),  # Orange
        (1.0, 0.0, 0.0),  # Red
    ]
    mapColor = LinearSegmentedColormap.from_list("WaterTreesMountains", colors)

    # Plot the results
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    im1 = axs.imshow(data_set1, vmin=-626.96027, vmax=778.96765, cmap=mapColor)
    axs.set_title("VAE Predictions")
    axs.set_yticklabels(reversed(axs.get_yticklabels()))
    axs.set_xlabel("X Coordinates")
    axs.set_ylabel("Y Coordinates")
    fig.colorbar(im1, label="Target Height (meters)", ax=axs, location="bottom")
    plt.show()
    plt.close()
