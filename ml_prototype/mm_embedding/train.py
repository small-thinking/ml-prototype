import os
import torch
import torch.nn as nn
from tqdm import tqdm
from ml_prototype.mm_embedding.data import create_dataloader, InMemoryDataset
from ml_prototype.mm_embedding.module import (
    ImageEncoder, TextEncoder, FusionLayer, MultimodalEmbeddingModel
)
from torchvision import transforms
import torch.nn.functional as F
from ml_prototype.mm_embedding.preprocess import load_images_as_batch
import wandb


# Step 1: Configuration
def get_config():
    return {
        "text_folder": "~/Downloads/multimodal/abo-listings",
        "image_folder": "~/Downloads/multimodal/images",
        "model": {
            "image_encoder": "google/vit-base-patch16-224-in21k",
            "text_encoder": "bert-base-multilingual-cased",
            "output_dim": 128,
        },
        "train": {
            "batch_size": 128,
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "image_transforms": transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to match model input
                transforms.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB
                transforms.ToTensor(),          # Convert to tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ]),
            "image_augments": transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to match model input
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        },
    }


# Step 2: Data Preparation
def prepare_data(config):
    """
    Prepares the dataloader for training.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        DataLoader: PyTorch DataLoader with InMemoryDataset.
    """
    dataset = InMemoryDataset(
        text_folder=os.path.expanduser(config["text_folder"]),
        image_folder=os.path.expanduser(config["image_folder"]),
    )
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True
    )
    return dataloader


# Step 3: Model Initialization
def initialize_model(config):
    device = config["train"]["device"]
    image_encoder = ImageEncoder(
        model_name=config["model"]["image_encoder"],
        output_dim=config["model"]["output_dim"],
        freeze=True,
    )
    text_encoder = TextEncoder(
        model_name=config["model"]["text_encoder"],
        output_dim=config["model"]["output_dim"],
        device=device,
    )
    fusion_layer = FusionLayer(
        input_dim=config["model"]["output_dim"] * 2,
        output_dim=config["model"]["output_dim"],
    )
    multimodal_model = MultimodalEmbeddingModel(image_encoder, text_encoder, fusion_layer, device=device)
    return multimodal_model


def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Computes the NT-Xent loss between two sets of embeddings.

    Args:
        z_i (torch.Tensor): Embeddings from original images. Shape: (N, D)
        z_j (torch.Tensor): Embeddings from augmented images. Shape: (N, D)
        temperature (float): Scaling factor.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # Shape: (2N, D)

    # Compute cosine similarity matrix
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # Shape: (2N, 2N)

    # Scale similarities by temperature
    sim = sim / temperature

    # Create labels for positive pairs
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)], dim=0).to(z.device)

    # Mask to exclude self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -9e15)

    # Compute cross-entropy loss
    loss = F.cross_entropy(sim, labels)

    # Normalize loss by batch size
    loss = loss / (2 * N)
    return loss


# Step 4: Training Loop
def train_model(model, dataloader, config):
    """
    Trains the multimodal embedding model with self-supervised learning.

    Args:
        model (nn.Module): The multimodal embedding model.
        dataloader (DataLoader): DataLoader with InMemoryDataset.
        config (dict): Configuration dictionary.
    """
    device = config["train"]["device"]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # Define the augmentation transform
    augmentation_transform = config["train"]["image_augments"]

    for epoch in range(config["train"]["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['num_epochs']}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            texts = batch['text']  # List of strings
            image_paths = batch['image_path']  # List of image file paths (strings)

            # Load original images
            original_images = load_images_as_batch(image_paths, config["train"]["image_transforms"]).to(device)

            # Load augmented images
            augmented_images = load_images_as_batch(image_paths, augmentation_transform).to(device)

            # Forward pass on original images
            embeddings_original = model(original_images, texts)      # Shape: (N, D)

            # Forward pass on augmented images
            embeddings_augmented = model(augmented_images, texts)    # Shape: (N, D)

            # Compute NT-Xent loss
            loss = nt_xent_loss(embeddings_original, embeddings_augmented, temperature=0.5)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute cosine similarity for monitoring
            cosine_sim = F.cosine_similarity(embeddings_original, embeddings_augmented).mean().item()

            # Update progress bar with loss and similarity
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Cosine Sim": f"{cosine_sim:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            # Log metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "batch_loss": loss.item(),
                "cosine_similarity": cosine_sim,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Print epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']} | Loss: {avg_loss:.4f}")
        # Log epoch-level metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss
        })

        # # Optionally, save model checkpoints
        # checkpoint_path = f"model_epoch_{epoch+1}.pth"
        # torch.save(model.state_dict(), checkpoint_path)
        # print(f"Saved model checkpoint at {checkpoint_path}")


# Step 5: Evaluation (Optional)
def evaluate_model(model, dataloader, config):
    device = config["train"]["device"]
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    criterion = nn.MSELoss()  # Replace with the appropriate evaluation metric

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            texts = batch["item_name"]

            embeddings = model(images, texts)

            # Dummy target for demonstration; replace with your actual target
            target = torch.randn_like(embeddings).to(device)

            loss = criterion(embeddings, target)
            total_loss += loss.item()

            # Update progress bar with current loss
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}"
            })

    # Print evaluation summary
    print(f"Evaluation Loss: {total_loss/len(dataloader):.4f}")


# Step 6: Main Function
def main():
    config = get_config()
    run_name = f"batch-{config['train']['batch_size']}"
    # Initialize W&B
    wandb.init(
        project="multimodal-embedding",  # Replace with your project name
        config=config,                    # Log all hyperparameters
        name=run_name,                    # Set run name
        reinit=True                       # Allows multiple runs in a single script
    )
    dataloader = prepare_data(config)
    model = initialize_model(config)
    train_model(model, dataloader, config)
    # Uncomment for evaluation
    # evaluate_model(model, dataloader, config)


if __name__ == "__main__":
    main()
