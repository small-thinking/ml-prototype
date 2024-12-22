"""This file contains the test code for data."""
import os
from ml_prototype.mm_embedding.data import SequentialAugmenter, create_dataloader
from augly.text import augmenters as text_aug
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
from ml_prototype.mm_embedding.data import InMemoryDataset


def load_data():
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

    dataset = InMemoryDataset(
        text_folder=text_folder,
        image_folder=image_folder,
        text_augment=None,
        image_augment=image_augment,
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=True
    )

    for idx, batch in enumerate(dataloader):
        print(batch)
        if idx > 10:
            break


if __name__ == "__main__":
    load_data()
