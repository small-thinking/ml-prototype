import torch
from torchvision import transforms
from PIL import Image
import os

def inspect_image_shape(image_path: str):
    # Define the transformations
    transform = transforms.Compose([
        # transforms.Resize((50, 50)),  # Resize the image to 50x50
        transforms.ToTensor(),         # Convert the image to a tensor
    ])

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    transformed_image = transform(image)

    # Print the shape of the transformed image
    print("Transformed image shape:", transformed_image.shape)

# Example usage
image_path = os.path.expanduser("~/Downloads/Pistachio_Image_Dataset/Pistachio_Image_Dataset/Siirt_Pistachio/siirt 1.jpg")
inspect_image_shape(image_path)