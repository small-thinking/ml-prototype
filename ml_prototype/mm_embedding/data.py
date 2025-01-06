# This file includes the data processing of the multi-modality item embedding learning.
from torch.utils.data import Dataset
from .preprocess import get_text_index, get_image_index, merge_text_image
from torch.utils.data import DataLoader
import torch
from torch.utils.data import random_split


class SequentialAugmenter:
    def __init__(self, augmenters):
        self.augmenters = augmenters

    def augment(self, text: str) -> str:
        for augmenter in self.augmenters:
            text = augmenter.augment(text)
        return text


class InMemoryDataset(Dataset):
    """Load the text and image data into memory and provide the dataset interface."""
    def __init__(self, text_folder: str, image_folder: str):
        self.text_dict = get_text_index(text_folder)
        self.image_dict = get_image_index(image_folder)
        self.data_dict: dict = merge_text_image(self.text_dict, self.image_dict)
        self.data = list(self.data_dict.values())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, image_path = self.data[idx]
        return {
            'text': text,
            'image_path': image_path
        }


def create_dataloaders(dataset, batch_size=32, shuffle=True, num_workers=8, val_split=0.2, seed=42):
    """
    Splits the dataset into training and validation sets and creates DataLoaders for each.
    Args:
        dataset (Dataset): PyTorch Dataset object.
        batch_size (int, optional): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data.
        num_workers (int, optional): Number of subprocesses for data loading.
        val_split (float, optional): Fraction of data to be used for validation.
        seed (int, optional): Random seed for reproducibility.
    Returns:
        dict: A dictionary containing 'train' and 'val' DataLoaders.
    """
    # Calculate the number of samples for validation
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # Ensure reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Dataset split into {train_size} training and {val_size} validation samples.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for validation
        num_workers=num_workers,
        pin_memory=True
    )

    return {
        'train': train_loader,
        'val': val_loader
    }
