import os
import torch
import torch.nn as nn
from tqdm import tqdm
from ml_prototype.mm_embedding.data import create_dataloader
from ml_prototype.mm_embedding.module import (
    ImageEncoder, TextEncoder, FusionLayer, MultimodalEmbeddingModel
)
from torchvision import transforms
from ml_prototype.mm_embedding.preprocess import load_images_as_batch


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
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "image_transforms": transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the image to the specified dimensions
                transforms.Lambda(lambda img: img.convert("RGB")),  # Convert grayscale images to RGB
                transforms.ToTensor(),      # Convert the image to a PyTorch tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
            ]),
        },
        "augmentations": {
            "resize": (224, 224),
            "horizontal_flip": True,
        },
    }


# Step 2: Data Preparation
def prepare_data(config):
    dataloader = create_dataloader(
        text_folder=os.path.expanduser(config["text_folder"]),
        image_folder=os.path.expanduser(config["image_folder"]),
        text_augment=None,  # Optional text augmentations
        image_augment=config["train"]["image_transforms"],
        batch_size=config["train"]["batch_size"],
        shuffle=True,
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


# Step 4: Training Loop
def train_model(model, dataloader, config):
    device = config["train"]["device"]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    criterion = nn.MSELoss()

    for epoch in range(config["train"]["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['train']['num_epochs']}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"]
            images_tensor = load_images_as_batch(images, config["train"]["image_transforms"]).to(device, dtype=torch.float32)
            texts = batch["item_name"]

            # Forward pass
            embeddings = model(images_tensor, texts)

            # Dummy target for demonstration; replace with your actual target
            target = torch.randn_like(embeddings).to(device)

            loss = criterion(embeddings, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Update progress bar with loss and learning rate
            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # Print epoch summary
        print(f"Epoch {epoch+1}/{config['train']['num_epochs']} | Loss: {epoch_loss/len(dataloader):.4f}")


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
    dataloader = prepare_data(config)
    model = initialize_model(config)
    train_model(model, dataloader, config)
    # Uncomment for evaluation
    # evaluate_model(model, dataloader, config)


if __name__ == "__main__":
    main()
