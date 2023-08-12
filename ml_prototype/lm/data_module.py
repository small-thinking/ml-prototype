"""Data module for language model.
"""

import os
from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset


class InMemoryDataset(Dataset):
    def __init__(self, lines: List[str], stoi: Dict[str, int], config: Dict[str, Any]):
        self.lines = lines
        self.num_samples = config.get("samples_per_epoch", 10000)
        self.stoi = stoi
        assert "context_size" in config, "context_size must be specified in config"
        self.context_size = config["context_size"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Random sample from lines by randoming chosing the start position, and sample a subsequence with length of context_size."""
        i = torch.randint(0, len(self.lines) - self.context_size - 1, (1,))
        x = torch.Tensor(
            [self.stoi[c] for c in self.lines[i : i + self.context_size]]
        ).long()
        y = torch.Tensor(
            [self.stoi[c] for c in self.lines[i + 1 : i + 1 + self.context_size]]
        ).long()
        assert x.shape == (
            self.context_size,
        ), f"x.shape: {x.shape}, expect: {self.context_size}"
        return x, y


class InMemoryDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_path = config["data_path"]
        self.batch_size = config["batch_size"]
        assert "data_path" in config, "data_path must be specified in config"

    def prepare_data(self):
        self.lines = open(os.path.expanduser(self.data_path)).read()
        vocab = sorted(list(set(self.lines)))
        self.itos = {i: word for i, word in enumerate(vocab)}
        self.stoi = {word: i for i, word in enumerate(vocab)}
        print("Data module: Vocab size: {}".format(len(vocab)))
        train_lines, val_lines = (
            self.lines[: int(len(self.lines) * 0.8)],
            self.lines[int(len(self.lines) * 0.8) :],
        )
        self.train_dataset = InMemoryDataset(train_lines, self.stoi, self.config)
        self.val_dataset = InMemoryDataset(val_lines, self.stoi, self.config)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
