"""Data module for language model.
"""

from glob import glob
import json
import os
from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset

from lm.tokenizer import Tokenizer, BytePairTokenizer


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
        assert x.shape == torch.Size([self.seq_len]), f"x.shape: {x.shape}, expected: {torch.Size([self.seq_len])}"
        return x, y


class InMemoryDataModule(pl.LightningDataModule):
    def __init__(self, config: Dict[str, Any], tokenizer: Tokenizer):
        super().__init__()
        self.config = config
        self.data_path = config["data_path"]
        self.batch_size = config["batch_size"]
        self.tokenizer = tokenizer
        assert "data_path" in config, "data_path must be specified in config"

    def prepare_data(self):
        self.text = ""
        if not os.path.isdir(self.data_path):
            print(f"Load data from {self.data_path}...")
            with open(os.path.expanduser(self.data_path), "r", encoding="utf-8") as f:
                # Strip for non-empty rows.
                rows = [row.strip() if len(row) >= 2 else row for row in f.readlines()]
                self.text += " ".join(rows)
        else:
            list_files = glob(os.path.join(self.data_path, "*.txt"))
            for file_path in list_files:
                print(f"Load data from {file_path}...")
                with open(os.path.expanduser(file_path), "r", encoding="utf-8") as f:
                    # Strip for non-empty rows.
                    rows = [row.strip() if len(row) >= 2 else row for row in f.readlines()]
                    self.text += " ".join(rows)

        print("Data module: Vocab size: {}".format(self.tokenizer.vocab_size()))
        train_text, val_text = (
            self.text[: int(len(self.text) * 0.8)],
            self.text[int(len(self.text) * 0.8) :],
        )
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
