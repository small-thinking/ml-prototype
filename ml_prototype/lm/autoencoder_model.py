"""
"""

import lightning.pytorch as pl
import lightning.pytorch.loggers.wandb as wandb
import torch
import torch.nn as nn
from ml_prototype.lm.autoencoder_module import AutoEncoder
from typing import Dict

class AutoEncoderModel(pl.LightningModule):
    def __init__(
        self,
        model: AutoEncoder,
        loss: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        
    def training_step(self, batch, batch_idx) -> Dict[str, float]:
        # Validate shape
        x = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> Dict[str, float]:
        x = batch
        x_hat = self.model(x)
        loss = self.loss(x_hat, x)
        self.log("val_loss", loss)
        return loss