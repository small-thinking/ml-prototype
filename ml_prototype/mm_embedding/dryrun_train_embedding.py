"""
This script tests the forward pass of the TabularEmbeddingModel using the DataLoader.
Run this script with the command:
    python -m ml_prototype.mm_embedding.dryrun_train_embedding --config ml_prototype/mm_embedding/config.yaml
"""
import torch
from torch.utils.data import DataLoader
from jsonargparse import ArgumentParser, ActionConfigFile
from ml_prototype.mm_embedding.config import MyConfig
from ml_prototype.mm_embedding.module import TabularEmbeddingModel
from ml_prototype.mm_embedding.util import Logger
from ml_prototype.mm_embedding.train_embedding import prepare_data, initialize_model
from tqdm import tqdm


def test_forward_pass(
    model: TabularEmbeddingModel, dataloader: DataLoader, logger: Logger, num_batches: int = 3
):
    """
    Test the forward pass of the model with a specified number of data batches.

    Args:
        model (TabularEmbeddingModel): The model to test.
        dataloader (DataLoader): DataLoader to fetch the data.
        config (MyConfig): Configuration object.
        logger (Logger): Logger instance.
        num_batches (int, optional): Number of batches to test. Defaults to 3.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model
    model.eval()  # Set model to evaluation mode
    logger.info(f"Testing forward pass on {num_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if batch_idx >= num_batches:
                break

            # Move batch data to the appropriate device
            tabular_features = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(tabular_features)

            # Log the output shape
            logger.info(f"Batch {batch_idx + 1}: Output shape: {outputs.shape}")

            # Optionally, print the outputs
            print(f"Batch {batch_idx + 1}:")
            print(outputs)
            print("-" * 50)


def main():
    logger = Logger(level="info")

    # Parse arguments and load configuration
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile, help="Path to the config file.")
    parser.add_class_arguments(MyConfig, 'my_config')
    args = parser.parse_args()

    config = parser.instantiate_classes(args).my_config
    logger.info(f"Configuration loaded: {config}")

    # Prepare data loaders
    dataloader = prepare_data(config, logger)
    # Initialize the model
    model = initialize_model(config, logger)
    # Test the forward pass
    test_forward_pass(model, dataloader, logger, num_batches=3)

    logger.info("Testing completed successfully.")


if __name__ == "__main__":
    main()
