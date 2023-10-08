"""Data module for language model.
"""
import json
import os
import random
from collections import deque
from glob import glob
from typing import Any, Deque, Dict, List

import lightning.pytorch as pl
import torch
from lm.tokenizer import BytePairTokenizer, Tokenizer
from torch.utils.data import ConcatDataset, Dataset


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
        self.data_folder = config["data_folder"]
        self.batch_size = config["batch_size"]
        self.tokenizer = tokenizer
        assert "data_folder" in config, "data_folder must be specified in config"

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
        text = ""
        with open(file_path, "r") as f:
            # Strip for non-empty rows.
            rows = [row.strip() if len(row) >= 2 else row for row in f.readlines()]
            text += " ".join(rows)
        self.tokens = tokenizer.encode(text)
        self.seq_len = config["seq_len"]
        text_len = len(self.tokens)
        # Sample based on the smaller of the the num_words / seq_len and the sample per epoch.
        self.num_samples = min(
            text_len // (self.seq_len // 5), config.get("samples_per_epoch", 10000)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
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

    def prepare_data(self):
        pass  # No specific preparation in this example

    def setup(self, stage=None):
        train_files = glob(os.path.join(self.data_folder, "train", "*.txt"))
        val_files = glob(os.path.join(self.data_folder, "val", "*.txt"))

        train_datasets = [
            SingleFileDataset(f, self.config, self.tokenizer) for f in train_files
        ]
        val_datasets = [
            SingleFileDataset(f, self.config, self.tokenizer) for f in val_files
        ]

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
