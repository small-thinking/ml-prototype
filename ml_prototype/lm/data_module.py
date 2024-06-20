"""Data module for language model.
"""
import datetime
import hashlib
import os
import random
from glob import glob
from typing import Any, Dict, List, Tuple, Optional

import lightning.pytorch as pl
import torch
from ml_prototype.lm.tokenizer import Tokenizer
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from PIL import Image


class InMemoryDataset(Dataset):
    def __init__(self, text: str, config: Dict[str, Any], tokenizer: Tokenizer):
        self.tokens = tokenizer.encode(text)
        self.num_samples = config.get("samples_per_epoch", 10000)
        assert "seq_len" in config, "seq_len must be specified in config"
        self.seq_len = config["seq_len"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Random sample from lines by randomly choosing the start position,
        and sample a subsequence with a length of seq_len.
        """
        i = torch.randint(0, len(self.tokens) - self.seq_len - 1, (1,)).item()
        x = self.tokens[i : i + self.seq_len]
        y = self.tokens[i + 1 : i + self.seq_len + 1]
        assert x.shape == torch.Size(
            [self.seq_len]
        ), f"x.shape: {x.shape}, expected: {torch.Size([self.seq_len])}"
        return x, y


class InMemoryDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        assert "data_folder" in config, "data_folder must be specified in config"
        self.data_folder = config["data_folder"]
        self.batch_size = config["batch_size"]
        self.tokenizer = tokenizer

    def _read_text_files(self, list_files: List[str]):
        text = ""
        for file_path in list_files:
            with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
                # Strip for non-empty rows.
                rows = [row.strip() if len(row) >= 2 else row for row in f.readlines()]
                text += " ".join(rows)
        return text

    def prepare_data(self):
        # Read train and val data from corresponding subfolders
        train_folder = os.path.join(self.data_folder, "train")
        val_folder = os.path.join(self.data_folder, "val")
        train_files = glob(os.path.join(train_folder, "**", "*.txt"), recursive=True)
        val_files = glob(os.path.join(val_folder, "**", "*.txt"), recursive=True)
        print(f"train files: {train_files}")
        print(f"val files: {val_files}")

        train_text = self._read_text_files(train_files)
        val_text = self._read_text_files(val_files)

        print("Data module: Vocab size: {}".format(self.tokenizer.vocab_size()))

        # Create train and val datasets
        self.train_dataset = InMemoryDataset(train_text, self.config, self.tokenizer)
        self.val_dataset = InMemoryDataset(val_text, self.config, self.tokenizer)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )


class SingleFileDataset(Dataset):
    def __init__(self, file_path: str, config: Dict[str, Any], tokenizer: Tokenizer):
        self.config = config
        self.file_path = file_path
        self.seq_len = config["seq_len"]
        self.tokenizer = tokenizer
        self.num_samples = self.config.get("samples_per_epoch", 10000)
        self.data_loaded = False
        self._load_data()

    def _load_data(self):
        text = ""
        with open(self.file_path, "r") as f:
            # Strip for non-empty rows.
            rows = [row.strip() if len(row) >= 2 else row for row in f.readlines()]
            text += " ".join(rows)
        self.tokens = self.tokenizer.encode(text)
        text_len = len(self.tokens)
        # Sample based on the smaller of the the num_words / seq_len and the sample per epoch.
        self.num_samples = min(
            text_len // (self.seq_len // 5), self.config.get("samples_per_epoch", 10000)
        )
        self.data_loaded = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # if not self.data_loaded:
        #     self._load_data()
        i = random.randint(0, len(self.tokens) - self.seq_len - 1)
        x = self.tokens[i : i + self.seq_len]
        y = self.tokens[i + 1 : i + self.seq_len + 1]
        return x, y


class ConcatDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data_folder = config["data_folder"]
        self.batch_size = config["batch_size"]

    def _assign_file_to_gpu(self, file_name: str, num_gpus: int):
        hash_val = int(hashlib.md5(file_name.encode()).hexdigest(), 16)
        return hash_val % num_gpus

    def prepare_data(self):
        pass  # No specific preparation in this example

    def setup(self, stage=None):
        train_files = glob(os.path.join(self.data_folder, "train", "*.txt"))
        val_files = glob(os.path.join(self.data_folder, "val", "*.txt"))

        train_datasets = []
        for i, file_path in enumerate(train_files):
            train_datasets.append(
                SingleFileDataset(file_path, self.config, self.tokenizer)
            )
            if i % 100 == 0:
                print(f"Initialized {i} train datasets...")
        val_datasets = []
        for i, file_path in enumerate(val_files):
            val_datasets.append(
                SingleFileDataset(file_path, self.config, self.tokenizer)
            )
            if i % 10 == 0:
                print(f"Initialized {i} val datasets.")

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )


class IncrementalDataset(Dataset):
    def __init__(
        self,
        file_list: List[Tuple[str, int]],
        config: Dict[str, Any],
        tokenizer: Tokenizer,
    ):
        self.config = config
        self.file_list = file_list
        self.seq_len = config["seq_len"]
        self.tokenizer = tokenizer
        self.current_file_idx = -1
        self.cur_file_sampled = -1
        # Loop through the list of files and calculate list of samples per epoch according to the file size.
        self.samples_per_file = []
        for i, (_, text_len) in enumerate(self.file_list):
            if i % 100 == 0:
                print(f"Scanned the metadata for {i} files...")
            self.samples_per_file.append(
                min(
                    text_len // (self.seq_len // 5),
                    self.config.get("samples_per_epoch", 10000),
                )
            )
        self.total_samples = sum(self.samples_per_file)

    def _load_data(self):
        # Load the content of a file according to the idx and reset the counters: current_file_idx, cur_file_sampled.
        self.current_file_idx += 1
        self.cur_file_sampled = 0
        text = ""
        with open(self.file_list[self.current_file_idx][0], "r") as f:
            # Strip for non-empty rows.
            rows = [row.strip() if len(row) >= 2 else row for row in f.readlines()]
            text += " ".join(rows)
        self.tokens = self.tokenizer.encode(text)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if (
            self.cur_file_sampled == -1
            or self.cur_file_sampled >= self.samples_per_file[self.current_file_idx]
        ):
            self._load_data()
        self.cur_file_sampled += 1
        i = random.randint(0, len(self.tokens) - self.seq_len - 1)
        x = self.tokens[i : i + self.seq_len]
        y = self.tokens[i + 1 : i + self.seq_len + 1]
        return x, y


class IncrementalLoadDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.data_folder = config["data_folder"]
        self.batch_size = config["batch_size"]

    def _assign_file_to_gpu(self, file_name: str, num_gpus: int):
        hash_val = int(hashlib.md5(file_name.encode()).hexdigest(), 16)
        return hash_val % num_gpus

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        start_time = datetime.datetime.now()
        # 1. Get the list of files in train and val subfolder, and also get their file sizes. Store in list of tuples.
        train_files = [
            (
                os.path.join(self.data_folder, "train", f),
                os.path.getsize(os.path.join(self.data_folder, "train", f)),
            )
            for f in os.listdir(os.path.join(self.data_folder, "train"))
        ]
        val_files = [
            (
                os.path.join(self.data_folder, "val", f),
                os.path.getsize(os.path.join(self.data_folder, "val", f)),
            )
            for f in os.listdir(os.path.join(self.data_folder, "val"))
        ]

        # # Assuming you have num_devices method somewhere to get the number of devices
        # num_devices = self.num_devices()

        # train_files = [
        #     f for f in train_files if self._assign_file_to_gpu(f[0], num_devices) == self.current_device_index()
        # ]
        # val_files = [f for f in val_files if self._assign_file_to_gpu(f[0], num_devices) == self.current_device_index()]

        self.train_dataset = IncrementalDataset(
            train_files, self.config, self.tokenizer
        )
        self.val_dataset = IncrementalDataset(val_files, self.config, self.tokenizer)
        end_time = datetime.datetime.now()
        delta = end_time - start_time
        print(f"Time spent in dataloading setup: {delta} seconds.")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )


class ImageFileDataset(Dataset):
    """Dataset for image files."""
    
    def __init__(self, folder_dir: str, transform: Optional[transforms.Compose], suffix: str = ".jpg"):
        self.folder_dir = folder_dir
        self.transform = transform
        self.suffix = suffix
        self.filenames = glob(os.path.join(folder_dir, f"*{suffix}"))
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])
        if self.transform:
            img = self.transform(img)
        return img
    
    
class ImageDataModule(pl.LightningDataModule):
    """Data module for image files."""
    
    def __init__(self, config: Dict[str, Any], transform: Optional[transforms.Compose]):
        super().__init__()
        self.config = config
        self.transform = transform
        self.data_folder = str(config["data_folder"])
        self.batch_size = config["batch_size"]
        
    def prepare_data(self):
        pass
        
    def setup(self):
        train_folder = os.path.join(self.data_folder, "train")
        val_folder = os.path.join(self.data_folder, "val")
        self.train_dataset = ImageFileDataset(train_folder, transform=self.transform)
        self.val_dataset = ImageFileDataset(val_folder, transform=self.transform)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )