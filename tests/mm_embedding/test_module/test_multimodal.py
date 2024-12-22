"""Test script for the multimodal embedding model."""

import os
from ml_prototype.mm_embedding.module import (
    ImageEncoder, TextEncoder, FusionLayer, MultimodalEmbeddingModel
)
from ml_prototype.mm_embedding.data import create_dataloader
from ml_prototype.mm_embedding.preprocess import load_images_as_batch
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from torchvision import transforms
from torchviz import make_dot


def test_multimodal_model():
    """Test the multimodal model with real data loaded from the dataloader."""

    # Mock directories for text and image data
    text_folder = os.path.expanduser("~/Downloads/multimodal/abo-listings")
    image_folder = os.path.expanduser("~/Downloads/multimodal/images")

    # Define image transformations
    resize: tuple = (224, 224)
    image_transforms = transforms.Compose([
        transforms.Resize(resize),  # Resize the image to the specified dimensions
        transforms.ToTensor(),      # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
    ])

    # Define image augmentations
    image_augment = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip()
    ])

    # Create dataloader
    dataloader = create_dataloader(
        text_folder=text_folder,
        image_folder=image_folder,
        text_augment=None,  # You can replace with text augmentations if needed
        image_augment=image_augment,
        batch_size=4,
        shuffle=True
    )

    # Initialize model components
    image_encoder = ImageEncoder(model_name='google/vit-base-patch16-224-in21k', output_dim=256, freeze=True)
    text_encoder = TextEncoder(model_name='bert-base-multilingual-cased', output_dim=256)
    fusion_layer = FusionLayer(input_dim=512, output_dim=128)
    multimodal_model = MultimodalEmbeddingModel(image_encoder, text_encoder, fusion_layer)

    # Print the model architecture
    print("\nModel Architecture:")
    print(multimodal_model)

    # Test the multimodal model
    for idx, batch in enumerate(dataloader):
        images = batch['image']  # Image tensors
        print(f"Batch {idx}: Loaded images: {images}")
        texts = batch['item_name']  # Text data (list of strings)

        image_tensor = load_images_as_batch(file_paths=images, image_transforms=image_transforms)

        # Forward pass through the model
        embeddings = multimodal_model(images=image_tensor, texts=texts)

        # Assertions and debugging
        assert embeddings.shape[0] == len(images), "Batch size mismatch"
        print(f"Batch {idx}: Generated multimodal embeddings shape: {embeddings.shape}")

        # Visualize the computation graph
        if idx == 0:  # Visualize only for the first batch
            print("\nGenerating computation graph...")
            graph = make_dot(embeddings, params=dict(multimodal_model.named_parameters()))
            graph.render("multimodal_model_graph", format="png", cleanup=True)
            print("Computation graph saved as multimodal_model_graph.png")

        # Test a few batches
        if idx >= 10:
            break


if __name__ == "__main__":
    test_multimodal_model()
