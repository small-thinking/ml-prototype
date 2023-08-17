"""The example code of tiny llama.
"""
from typing import Any, Dict

import lightning.pytorch as pl
import torch
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


class TorchScriptCallback(pl.Callback):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.context_size = config["context_size"]
        self.vocab_size = config["vocab_size"]

    def on_fit_end(self, trainer, pl_module):
        model = (
            pl_module.model
        )  # Assuming the actual model is stored in 'model' attribute
        example_input = torch.randint(
            0, self.vocab_size, (self.batch_size, self.context_size), dtype=torch.long
        )
        scripted_model = torch.jit.trace(model, example_input)
        print("Save torch script model...")
        torch.jit.save(scripted_model, f"model.pt")
        print("Torch script model saved.")


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
