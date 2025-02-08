from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from zenml import step


@step
def evaluate_vae(vae: VAE, data: dict) -> dict:
    """
    Evaluate the VAE model on the test dataset.

    Args:
        vae (VAE): Trained VAE model.
        data (dict): Dictionary containing test data.
            Expected keys: "x_test", "y_test", "scaler_Y".

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Predict on test data
    _, _, y_pred_test = vae.encoder.predict(data["x_test"])
    reconstructed_predictions = vae.decoder.predict(y_pred_test)

    # Inverse transform predictions
    y_test_full_range = data["scaler_Y"].inverse_transform(
        data["y_test"].reshape(-1, 1)
    )
    test_predict_full_scale = data["scaler_Y"].inverse_transform(
        reconstructed_predictions
    )

    # Calculate metrics
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test_full_range, test_predict_full_scale)),
        "RMSPE": rmspe(y_test_full_range, test_predict_full_scale),
        "MAE": mean_absolute_error(y_test_full_range, test_predict_full_scale),
        "MAPE": mean_absolute_percentage_error(
            y_test_full_range, test_predict_full_scale
        ),
        "R^2": r2_score(y_test_full_range, test_predict_full_scale),
    }

    # Print metrics
    print("Test stats:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics
