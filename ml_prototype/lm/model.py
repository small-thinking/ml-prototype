"""The example code of tiny llama.
"""
from typing import Dict

import lightning.pytorch as pl
import torch.nn as nn
from lm.module import LanguageModule


class DummyCallback(pl.Callback):
    """Used to keep the display of epoch in the command line."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = pl_module.model
        print(model)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        print("")


class Seq2SeqLM(pl.LightningModule):
    def __init__(self, model: LanguageModule, loss: nn.Module, vocab_size: int):
        super().__init__()
        self.model = model
        self.loss = loss
        self.vocab_size = vocab_size

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx) -> Dict[str, float]:
        x, y = batch
        config = self.model.config
        assert y.shape[1:] == (
            config["context_size"],
        ), f"Training_step: y.shape: {y.shape}, expect: {-1, config['context_size']}"
        y_hat = self.model(x)
        loss = self.loss(
            y_hat.view(-1, self.vocab_size),
            y.view(-1),
        )
        metric_dict = {"loss": loss}
        self.log_dict(metric_dict, prog_bar=True, on_epoch=True, on_step=False)
        return metric_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat.view(-1, self.vocab_size), y.view(-1))
        metric_dict = {"val_loss": loss}
        self.log_dict(metric_dict, prog_bar=True, on_epoch=True, on_step=False)
        return metric_dict
