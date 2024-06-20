
"""Run the following command to run the tests:
    poetry run pytest tests/lm/test_data_module.py
"""

import os
import pytest
import tempfile
from PIL import Image
import torch
from torchvision import transforms
from ml_prototype.lm.data_module import ImageFileDataset, ImageDataModule


class TestImageDataset:
    @pytest.fixture(scope="class")
    def setup_test_dir(self):
        # Create a temporary directory
        test_dir = tempfile.TemporaryDirectory()

        # Create some dummy images
        for i in range(5):
            image = Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
            image.save(os.path.join(test_dir.name, f'test_image_{i}.jpg'))

        yield test_dir

        # Cleanup the temporary directory
        test_dir.cleanup()

    @pytest.fixture(scope="class")
    def setup_transform(self):
        # Define a simple transform
        return transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])

    def test_len(self, setup_test_dir):
        dataset = ImageFileDataset(setup_test_dir.name, transform=None, suffix="jpg")
        assert len(dataset) == 5

    def test_getitem(self, setup_test_dir, setup_transform):
        dataset = ImageFileDataset(setup_test_dir.name, transform=setup_transform, suffix="jpg")
        image = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 50, 50)  # Check if the transform was applied

    def test_no_transform(self, setup_test_dir):
        dataset = ImageFileDataset(setup_test_dir.name, transform=None, suffix="jpg")
        image = dataset[0]
        assert isinstance(image, Image.Image)  # Check if the image is a PIL image without transform

    def test_empty_directory(self):
        empty_dir = tempfile.TemporaryDirectory()
        dataset = ImageFileDataset(empty_dir.name, transform=None, suffix="jpg")
        assert len(dataset) == 0
        empty_dir.cleanup()

    def test_invalid_index(self, setup_test_dir):
        dataset = ImageFileDataset(setup_test_dir.name, transform=None, suffix="jpg")
        with pytest.raises(IndexError):
            dataset[len(dataset)]  # Accessing out of bounds should raise an error


class TestImageDataModule:
    @pytest.fixture(scope="class")
    def setup_test_dirs(self):
        # Create temporary directories for train and val
        train_dir = tempfile.TemporaryDirectory()
        val_dir = tempfile.TemporaryDirectory()

        # Create some dummy images in train and val directories
        for i in range(5):
            image = Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
            image.save(os.path.join(train_dir.name, f'train_image_{i}.jpg'))
            image.save(os.path.join(val_dir.name, f'val_image_{i}.jpg'))

        yield {"train_dir": train_dir, "val_dir": val_dir}

        # Cleanup the temporary directories
        train_dir.cleanup()
        val_dir.cleanup()

    @pytest.fixture(scope="class")
    def setup_transform(self):
        # Define a simple transform
        return transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor()
        ])

    @pytest.fixture(scope="class")
    def setup_config(self, setup_test_dirs):
        # Define the configuration for ImageDataModule
        return {
            "data_folder": {"train": setup_test_dirs["train_dir"].name, "val": setup_test_dirs["val_dir"].name},
            "batch_size": 2
        }

    def test_train_dataloader(self, setup_config, setup_transform):
        datamodule = ImageDataModule(config=setup_config, transform=setup_transform)
        datamodule.setup()

        train_dataloader = datamodule.train_dataloader()
        for batch in train_dataloader:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape == (setup_config["batch_size"], 3, 50, 50)  # Check batch size and image shape
            break  # Only check the first batch

    def test_val_dataloader(self, setup_config, setup_transform):
        datamodule = ImageDataModule(config=setup_config, transform=setup_transform)
        datamodule.setup()

        val_dataloader = datamodule.val_dataloader()
        for batch in val_dataloader:
            assert isinstance(batch, torch.Tensor)
            assert batch.shape == (setup_config["batch_size"], 3, 50, 50)  # Check batch size and image shape
            break  # Only check the first batch

    def test_len_train_dataset(self, setup_config, setup_transform):
        datamodule = ImageDataModule(config=setup_config, transform=setup_transform)
        datamodule.setup()

        assert len(datamodule.train_dataset) == 5  # Check number of images in the train dataset

    def test_len_val_dataset(self, setup_config, setup_transform):
        datamodule = ImageDataModule(config=setup_config, transform=setup_transform)
        datamodule.setup()

        assert len(datamodule.val_dataset) == 5  # Check number of images in the val dataset