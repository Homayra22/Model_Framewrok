from zenml import pipeline
from steps.data_loader import load_data
from steps.train import train_vae
from steps.evaluate import evaluate_vae
from steps.visualize import visualize_results


@pipeline
def vae_pipeline():
    """ZenML pipeline for training and evaluating the VAE model."""
    # Load data
    data = load_data()

    # Train the VAE model
    vae = train_vae(data)

    # Evaluate the VAE model
    metrics = evaluate_vae(vae, data)

    # Visualize results
    visualize_results(vae, data)


# Run the pipeline
if __name__ == "__main__":
    vae_pipeline()
