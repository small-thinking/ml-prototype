"""The example code of tiny llama.
"""
import os
from typing import Any, Dict

import lightning.pytorch as pl
import lightning.pytorch.loggers.wandb as wandb
import torch
import torch.nn as nn
from lm.module import LanguageModule


class DummyCallback(pl.Callback):
    """Used to keep the display of epoch in the command line."""

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        model = pl_module.model
        print(model)

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        optimizer = trainer.optimizers[0]


class CustomWandbLogger(wandb.WandbLogger):
    def __init__(self, name: str, project: str, config: Dict[str, Any], **kwargs):
        batch_size, seq_len, embed_dim = (
            config["batch_size"],
            config["seq_len"],
            config["embed_dim"],
        )
        num_heads, num_layers, dropout_ratio = (
            config["num_heads"],
            config["num_layers"],
            config["dropout_ratio"],
        )
        experiment_name = f"dummy-b{batch_size}-t{seq_len}-d{embed_dim}-h{num_heads}-l{num_layers}-dp{dropout_ratio}"
        super().__init__(name=name, project=project, experiment=experiment_name)


class TorchScriptCallback(pl.Callback):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.batch_size = config["batch_size"]
        self.seq_len = config["seq_len"]
        self.vocab_size = config["vocab_size"]
        self.save_every_epoch = config.get("save_every_epoch", 5)

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        # Save model every x epochs
        if (current_epoch + 1) % self.save_every_epoch == 0:
            model = (
                pl_module.model
            )  # Assuming the actual model is stored in 'model' attribute

            try:
                scripted_model = torch.jit.script(model)
            except Exception as e:
                print(f"Failed to script the model: {e}")
                return

            # Save the model with a filename that includes the current epoch
            model_name = f"model_epoch_{current_epoch + 1}.pt"
            print(f"Saving TorchScript model at epoch {current_epoch + 1}...")
            if os.path.exists(model_name):
                os.remove(model_name)
            torch.jit.save(scripted_model, model_name)
            print(f"TorchScript model saved as {model_name}.")

            # Remove models that are too old.
            old_model_name = (
                f"model_epoch_{(current_epoch + 1) - 3 * self.save_every_epoch}.pt"
            )
            if os.path.exists(old_model_name):
                print(f"Deleting old TorchScript model {old_model_name}...")
                os.remove(old_model_name)


class Seq2SeqLM(pl.LightningModule):
    def __init__(
        self,
        model: LanguageModule,
        loss: nn.Module,
        vocab_size: int,
        checkpoint_dir: str = None,
        checkpoint_epoch: str = None,
        lr_schedule_interval: str = None,
    ):
        super().__init__()

        if checkpoint_dir is not None:
            if checkpoint_epoch is not None:
                # Load model from a specific checkpoint
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"epoch={checkpoint_epoch}.ckpt"
                )
            else:
                # Load the latest checkpoint
                list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
                checkpoint_path = max(list_of_files, key=os.path.getctime)

            self.model = Seq2SeqLM.load_from_checkpoint(checkpoint_path).model
        else:
            self.model = model

        self.lr_schedule_interval = lr_schedule_interval
        self.loss = loss
        self.vocab_size = vocab_size

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx) -> Dict[str, float]:
        x, y = batch
        config = self.model.config
        seq_len = config["seq_len"]

        # Ensure the shape of y is as expected
        assert y.shape[1:] == (
            config["seq_len"],
        ), f"Training_step: y.shape: {y.shape}, expect: {-1, config['seq_len']}"

        # Forward pass
        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        y_hat = self.model(x, attn_mask)

        # Ensure the shape of y_hat is as expected
        assert y_hat.shape == torch.Size(
            [x.shape[0], config["seq_len"], self.vocab_size]
        ), f"y_hat.shape: {y_hat.shape}, expect: {torch.Size([x.shape[0], config['seq_len'], self.vocab_size])}"

        # Reshape for the loss function
        y = y.view(-1)  # [batch_size * seq_len]
        y_hat = y_hat.view(-1, self.vocab_size)  # [batch_size * seq_len, vocab_size]
        loss = self.loss(y_hat, y)

        # Log metrics
        metric_dict = {"loss": loss}
        self.log_dict(metric_dict, prog_bar=True, on_epoch=True, on_step=True)

        return metric_dict

    def validation_step(self, batch, batch_idx):
        x, y = batch
        config = self.model.config
        seq_len = config["seq_len"]

        attn_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        y_hat = self.model(x, attn_mask)

        # Reshape for the loss function
        y = y.view(-1)  # [batch_size * seq_len
        y_hat = y_hat.view(-1, self.vocab_size)  # [batch_size * seq_len, vocab_size]
        loss = self.loss(y_hat, y.view(-1))
        metric_dict = {"val_loss": loss}
        self.log_dict(metric_dict, prog_bar=True, on_epoch=True, on_step=True)
        return metric_dict

    def on_fit_start(self):
        """Update the lr scheduler for step."""
        if self.lr_schedulers and self.lr_schedule_interval == "step":
            for lr_scheduler_config in self.trainer.lr_scheduler_configs:
                lr_scheduler_config.interval = "step"
