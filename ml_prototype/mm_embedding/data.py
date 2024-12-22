# This file includes the data processing of the multi-modality item embedding learning.
from torch.utils.data import Dataset
from .preprocess import get_text_index, get_image_index, merge_text_image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from PIL import Image


class SequentialAugmenter:
    def __init__(self, augmenters):
        self.augmenters = augmenters

    def augment(self, text: str) -> str:
        for augmenter in self.augmenters:
            text = augmenter.augment(text)
        return text


class InMemoryDataset(Dataset):
    """Load the text and image data into memory and provide the dataset interface."""
    """Load the text and image data into memory with optional augmentations."""
    def __init__(
        self,
        text_folder: str,
        image_folder: str,
        text_augment: SequentialAugmenter | None = None,
        image_augment: Compose | None = None,
    ):
        self.text_dict = get_text_index(text_folder)
        self.image_dict = get_image_index(image_folder)
        self.data_dict = merge_text_image(self.text_dict, self.image_dict)
        self.data = list(self.data_dict.values())
        self.text_augment = text_augment
        self.image_augment = image_augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, any]:
        item_name, image_file = self.data[idx]
        # Apply text augmentation if provided
        if self.text_augment:
            item_name = self.text_augment.augment(item_name)
        image = Image.open(image_file)
        # Apply image augmentation if provided
        if self.image_augment:
            image = self.image_augment(image)

        return {"item_name": item_name, "image": image_file}


def create_dataloader(
    text_folder: str,
    image_folder: str,
    text_augment: SequentialAugmenter | None = None,
    image_augment: Compose | None = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = InMemoryDataset(
        text_folder=text_folder,
        image_folder=image_folder,
        text_augment=text_augment,
        image_augment=image_augment,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
