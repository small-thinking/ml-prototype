"""This module contains the implementation of the autoencoder module."""
import abc
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler

class AutoEncoderLLM(nn.Module):
    """Autoencoder model for the LLM.
    """
    def __init__(self, config: Dict[str, Any]):
        super(AutoEncoderLLM, self).__init__()
        self.config = config
        self.layer_sizes = self.config["layer_sizes"]
        self.dropout = self.config["dropout"]
        self.activation_type = self.config["activation_type"]
        self.net = self._build_model(
            layer_sizes=self.layer_sizes,
            dropout=self.dropout,
            activation_type=self.activation_type
        )
        
    def _build_model(
        self,
        layer_sizes: List[int],
        dropout: float,
        activation_type: str
    ) -> nn.Module:
        layers = []
        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            layers.append(nn.Dropout(dropout))
            if i < len(layer_sizes) - 1:
                if activation_type == "relu":
                    layers.append(nn.ReLU())
                elif activation_type == "tanh":
                    layers.append(nn.Tanh())
                elif activation_type == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Invalid activation type: {activation_type}")
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flat the input tensor from (batch_size, C, H, W) to (batch_size, C*H*W)
        x = x.view(x.size(0), -1)
        return self.net(x)
        
    
