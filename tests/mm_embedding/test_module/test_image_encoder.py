import torch
from ml_prototype.mm_embedding.module import ImageEncoder
import torchvision.transforms as transforms
from ml_prototype.mm_embedding.preprocess import load_images_as_batch
import os


def test_image_encoder():
    """Test the ImageEncoder class with dummy image data and real file paths."""

    image_dir = os.path.expanduser("~/Downloads/multimodal/images")
    image_file_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_file_paths.append(os.path.join(root, file))
    print(f"Found {len(image_file_paths)} image files in the folder.")
    # Only keep 10 images for testing
    image_file_paths = image_file_paths[:10]

    # Load images as batch tensor [B, C, H, W]
    resize: tuple = (224, 224)
    image_transforms = transforms.Compose([
        transforms.Resize(resize),  # Resize the image to the specified dimensions
        transforms.ToTensor(),      # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
    ])
    batch_images = load_images_as_batch(image_file_paths, image_transforms)
    print(f"Batch Images Shape: {batch_images.shape}")  # Expected: [B, 3, 224, 224]

    # Test Vision Transformer (ViT)
    print("\nTesting ImageEncoder with ViT-base-patch16-224-in21k:")
    image_encoder_vit = ImageEncoder(model_name='google/vit-base-patch16-224-in21k', output_dim=256)
    image_encoder_vit.eval()  # Set the encoder to evaluation mode

    with torch.no_grad():
        embeddings_vit = image_encoder_vit(batch_images)
    print(f"ViT Embedding Shape: {embeddings_vit.shape}")  # Expected: (batch_size, 256)

    # Test ResNet
    print("\nTesting ImageEncoder with ResNet-50:")
    image_encoder_resnet = ImageEncoder(model_name='microsoft/resnet-50', output_dim=256)
    image_encoder_resnet.eval()

    with torch.no_grad():
        embeddings_resnet = image_encoder_resnet(batch_images)
    print(f"ResNet Embedding Shape: {embeddings_resnet.shape}")  # Expected: (batch_size, 256)


if __name__ == "__main__":
    test_image_encoder()
