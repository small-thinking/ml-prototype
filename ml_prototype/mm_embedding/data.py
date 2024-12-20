# This file includes the data processing of the multi-modality item embedding learning.
from torch.utils.data import Dataset
from preprocess import get_text_index, get_image_index, merge_text_image
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import os
from typing import Callable
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from augly.text import augmenters as text_aug


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
        if self.text_augment_fn:
            item_name = self.text_augment.augment(item_name)

        # Apply image augmentation if provided
        if self.image_augment_fn:
            image_file = self.image_augment(image_file)

        return {"item_name": item_name, "image": image_file}


class InMemoryDataModule(pl.LightningDataModule):
    """DataModule for the multi-modality item embedding learning."""
    def __init__(
        self, text_folder: str, text_augment: SequentialAugmenter,
        image_folder: str, image_augment: Compose, batch_size: int = 32
    ):
        super().__init__()
        self.text_folder = text_folder
        self.text_augment = text_augment
        self.image_folder = image_folder
        self.image_augment = image_augment
        self.batch_size = batch_size

    def setup(self, stage: str | None = None):
        self.dataset = InMemoryDataset(self.text_folder, self.image_folder)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    # Define mock directories for text and image data
    text_folder = os.path.expanduser("~/Downloads/multimodal/abo-listings")
    image_folder = os.path.expanduser("~/Downloads/multimodal/images")

    # Implement the augmentations
    text_augmenters = [
        text_aug.TypoAugmenter(
            min_char=1, aug_char_min=1, aug_char_max=3, aug_char_p=0.1, aug_word_min=100,
            aug_word_max=1000, aug_word_p=0.1, typo_type="charmix",
            misspelling_dict_path=None, max_typo_length=1, priority_words=None,
        ),
    ]
    text_augment = SequentialAugmenter(augmenters=text_augmenters)
    image_augment = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip()
    ])

    # Create an instance of the DataModule
    data_module = InMemoryDataModule(
        text_folder=text_folder, image_folder=image_folder, batch_size=4,
        text_augment=text_augment, image_augment=image_augment
    )

    # Setup the DataModule (loads the data into memory)
    data_module.setup()

    # Get the DataLoader
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Iterate through the DataLoader to see the output
    print("Training DataLoader:")
    for batch in train_loader:
        print(batch)

    print("\nValidation DataLoader:")
    for batch in val_loader:
        print(batch)
