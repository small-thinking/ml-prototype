import os
import torch
import torch.nn as nn
from tqdm import tqdm
from ml_prototype.mm_embedding.data import create_dataloaders, InMemoryDataset
from ml_prototype.mm_embedding.module import (
    ImageEncoder, TextEncoder, FusionLayer, MultimodalEmbeddingModel
)
from torchvision import transforms
import torch.nn.functional as F
from ml_prototype.mm_embedding.preprocess import load_images_as_batch
import wandb


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent Loss).

    This loss is used in contrastive learning frameworks to maximize similarity
    between positive pairs and minimize similarity between negative pairs.
    """
    def __init__(self, temperature: float = 0.5, device: str = 'cpu'):
        """
        Initializes the NTXentLoss module.

        Args:
            temperature (float): The temperature scaling factor.
            device (str): The device to perform computations on ('cpu' or 'cuda').
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Computes the NT-Xent loss between two sets of embeddings.

        Args:
            z_i (torch.Tensor): Embeddings from original samples. Shape: (n, d)
            z_j (torch.Tensor): Embeddings from augmented samples. Shape: (n, d)

        Returns:
            torch.Tensor: The computed NT-Xent loss.
        """
        # Normalize the embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate the embeddings
        z = torch.cat([z_i, z_j], dim=0)  # Shape: (2n, d)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T)  # Shape: (2n, 2n)

        # Apply temperature scaling
        sim_matrix = sim_matrix / self.temperature

        # Create labels
        n = z_i.shape[0]
        labels = torch.arange(n).to(self.device)
        labels = torch.cat([labels + n, labels]).to(self.device)  # Positive pairs are shifted by n

        # Mask to remove similarity of samples to themselves
        mask = torch.eye(2 * n, dtype=torch.bool).to(self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# Step 1: Configuration
def get_config():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        print("Warning: Running on CPU. Consider using a GPU for faster training.")
    return {
        "text_folder": "~/Downloads/multimodal/abo-listings",
        "image_folder": "~/Downloads/multimodal/images",
        "wandb_log": True,
        "model": {
            "image_encoder": "google/vit-base-patch16-224-in21k",
            "text_encoder": "bert-base-multilingual-cased",
            "output_dim": 128,
        },
        "train": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 2,
            "val_split": 0.2,
            "seed": 42,
            "device": device,
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
    dataloaders = create_dataloaders(
        dataset=dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        val_split=config["train"]["val_split"],
        seed=config["train"]["seed"],
    )
    return dataloaders


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
    nt_xent_loss = NTXentLoss(temperature=0.5, device=device)

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
            loss = nt_xent_loss(embeddings_original, embeddings_augmented)

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
            if config["wandb_log"]:
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
        if config["wandb_log"]:
            # Log epoch-level metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "avg_loss": avg_loss
            })

        # Optionally, save model checkpoints
        checkpoint_path = f"model_epoch_{epoch+1}.ckpt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


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
    if config["wandb_log"]:
        # Log evaluation loss to W&B
        wandb.log({
            "evaluation_loss": total_loss / len(dataloader)
        })


# Step 6: Main Function
def main():
    config = get_config()
    run_name = f"batch-{config['train']['batch_size']}"
    if config["wandb_log"]:
        # Initialize W&B
        wandb.init(
            project="multimodal-embedding",  # Replace with your project name
            config=config,                    # Log all hyperparameters
            name=run_name,                    # Set run name
            reinit=True                       # Allows multiple runs in a single script
        )

    dataloaders = prepare_data(config)
    model = initialize_model(config)
    train_model(model, dataloaders["train"], config)
    # Uncomment for evaluation
    # evaluate_model(model, dataloaders["val"], config)
    if config["wandb_log"]:
        # Save the final model
        torch.save(model.state_dict(), "multimodal_embedding.pth")
        wandb.save("multimodal_embedding.pth")
        # Finish W&B run
        wandb.finish()


if __name__ == "__main__":
    main()
