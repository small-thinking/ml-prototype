"""This file contains the implementation of training the TabularEmbeddingModel.

Run this script with the command:
    python -m ml_prototype.mm_embedding.train_embedding --config ml_prototype/mm_embedding/config.yaml
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from jsonargparse import ArgumentParser, ActionConfigFile
from torch.utils.data import DataLoader
from ml_prototype.mm_embedding.module import TabularEmbeddingModel
from ml_prototype.mm_embedding.dataloader import create_dataloaders
from ml_prototype.mm_embedding.contrastive import ContrastiveLoss
from ml_prototype.mm_embedding.config import MyConfig
import copy
from typing import Any
from tqdm import tqdm
import wandb


def data_augmentation(batch: dict[str, torch.Tensor], augment_config: dict[str, Any]) -> dict[str, torch.Tensor]:
    """
    Apply data augmentation to the batch.

    Args:
        batch (dict[str, torch.Tensor]): Original batch of data.
        config (dict[str, Any]): Configuration for data augmentation.

    Returns:
        dict[str, torch.Tensor]: Augmented batch of data.
    """
    augmented_batch = copy.deepcopy(batch)

    # Apply Gaussian noise to numerical features
    if augment_config.augment_numerical:
        noise_std = augment_config.noise_std
        for key, value in augmented_batch.items():
            if value.dtype == torch.float32:
                augmented_batch[key] += torch.randn_like(value) * noise_std

    # Apply random changes to categorical features
    if augment_config.augment_categorical:
        change_prob = augment_config.change_prob
        change_delta = augment_config.change_delta
        for key, value in augmented_batch.items():
            if value.dtype == torch.int64:
                mask = torch.rand_like(value.float()) < change_prob
                new_val = value[mask] + torch.randint(-change_delta, change_delta + 1, value[mask].shape)
                new_val = torch.clamp(new_val, min=0)
                augmented_batch[key][mask] = new_val % value.max()

    return augmented_batch


def cosine_similarity(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity between two sets of embeddings.

    Args:
        z_i (torch.Tensor): Embeddings of the original data. Shape: [batch_size, embedding_dim]
        z_j (torch.Tensor): Embeddings of the augmented data. Shape: [batch_size, embedding_dim]

    Returns:
        torch.Tensor: Cosine similarity. Shape: [batch_size]
    """
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    return torch.sum(z_i * z_j, dim=1)


def train(
    model: TabularEmbeddingModel, dataloaders: dict[str, DataLoader],
    device: str, config: MyConfig,
):
    """
    Train the model using contrastive learning.

    Args:
        model (TabularEmbeddingModel): The model to train.
        dataloaders (dict[str, DataLoader]): Dictionary containing 'train' and 'val' DataLoaders.
        epochs (int): Number of training epochs.
        device (str): Device to run the training on.
        augment_config (dict): Configuration for data augmentation.
        use_wandb (bool): Whether to use weights and biases for tracking.
    """
    model.to(device)
    training_config = config.training_config
    augment_config = training_config.augment_config
    optimizer = optim.Adam(model.parameters(), lr=training_config.learning_rate)
    contrastive_loss_fn = ContrastiveLoss().to(device)
    epochs = training_config.epochs
    lr = training_config.learning_rate
    num_layers = config.model_config.transformer_layers
    emb_dim = config.model_config.embedding_dim
    ff_dim = config.model_config.ff_dim
    batch_size = config.data_module.batch_size

    if training_config.use_wandb:
        run_name = f"l_{num_layers}_emb_{emb_dim}_lr_{lr}_bs_{batch_size}"
        wandb.init(
            project="tabular-embedding",
            config={
                "epochs": epochs, "learning_rate": lr, "num_layers": num_layers,
                "embedding_dim": emb_dim, "ff_dim": ff_dim,
                "batch_size": batch_size
            },
            name=run_name
        )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        avg_similarity = 0.0
        all_similarities = []
        all_norms = []
        # Train the model in each epoch
        with tqdm(total=len(dataloaders["train"]), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in dataloaders["train"]:
                batch = {key: value.to(device) for key, value in batch.items()}
                augmented_batch = data_augmentation(batch, augment_config)

                # Forward pass for original and augmented data
                z_i = model(batch)
                z_j = model(augmented_batch)

                # Compute contrastive loss
                loss = contrastive_loss_fn(z_i, z_j)

                # Compute cosine similarity
                similarity = cosine_similarity(z_i, z_j)
                avg_similarity += similarity.mean().item()
                all_similarities.extend(similarity.detach().cpu().numpy())

                # Compute embedding norms
                norm_i = torch.norm(z_i, dim=1).mean().item()
                norm_j = torch.norm(z_j, dim=1).mean().item()
                all_norms.append((norm_i + norm_j) / 2)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                pbar.set_postfix({"loss": loss.item(), "similarity": similarity.mean().item()})
                pbar.update(1)

                if training_config.use_wandb:
                    wandb.log({"batch_loss": loss.item(), "batch_similarity": similarity.mean().item()})

        avg_loss = total_loss / len(dataloaders["train"])
        avg_similarity /= len(dataloaders["train"])
        avg_norm = sum(all_norms) / len(all_norms)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Avg Similarity: {avg_similarity:.4f}, Avg Norm: {avg_norm:.4f}")

        if training_config.use_wandb:
            wandb.log({
                "epoch_train_loss": avg_loss,
                "epoch_train_similarity": avg_similarity,
                "epoch_train_norm": avg_norm,
                "train_similarity_distribution": wandb.Histogram(all_similarities)
            })

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_similarity = 0.0
        val_all_similarities = []
        val_all_norms = []
        with torch.no_grad():
            with tqdm(total=len(dataloaders["val"]), desc=f"Validation {epoch+1}/{epochs}") as pbar:
                for batch in dataloaders["val"]:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    augmented_batch = data_augmentation(batch, augment_config)

                    # Forward pass for original and augmented data
                    z_i = model(batch)
                    z_j = model(augmented_batch)

                    # Compute contrastive loss
                    loss = contrastive_loss_fn(z_i, z_j)

                    # Compute cosine similarity
                    similarity = cosine_similarity(z_i, z_j)
                    val_similarity += similarity.mean().item()
                    val_all_similarities.extend(similarity.cpu().numpy())

                    # Compute embedding norms
                    norm_i = torch.norm(z_i, dim=1).mean().item()
                    norm_j = torch.norm(z_j, dim=1).mean().item()
                    val_all_norms.append((norm_i + norm_j) / 2)

                    val_loss += loss.item()

                    pbar.set_postfix({"val_loss": loss.item(), "val_similarity": similarity.mean().item()})
                    pbar.update(1)

        val_loss /= len(dataloaders["val"])
        val_similarity /= len(dataloaders["val"])
        val_avg_norm = sum(val_all_norms) / len(val_all_norms)
        print(
            f"Validation Loss: {val_loss:.4f}, "
            "Validation Similarity: {val_similarity:.4f}, "
            "Validation Norm: {val_avg_norm:.4f}"
        )

        if training_config.use_wandb:
            wandb.log({
                "epoch_val_loss": val_loss,
                "epoch_val_similarity": val_similarity,
                "epoch_val_norm": val_avg_norm,
                "val_similarity_distribution": wandb.Histogram(val_all_similarities)
            })

    if training_config.use_wandb:
        wandb.finish()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        action=ActionConfigFile, help='Path to YAML config file',
    )
    parser.add_class_arguments(MyConfig, 'my_config')
    parser.add_argument('--use_wandb', action='store_true', help='Use weights and biases for tracking')
    args = parser.parse_args()

    # Load configuration
    config = parser.instantiate_classes(args).my_config
    # Create DataLoaders with drop_last=True to drop the last incomplete batch
    dataloaders = create_dataloaders(data_module_config=config.data_module)
    # Initialize the model
    model = TabularEmbeddingModel(config=config)
    # Train the model
    train(model, dataloaders, device=config.device, config=config)


if __name__ == "__main__":
    main()
