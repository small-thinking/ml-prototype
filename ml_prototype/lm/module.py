"""
"""
import abc
from typing import Any, Dict, List

import torch
import torch.nn as nn


class LanguageModule(nn.Module, abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config


class FeedForwardModel(LanguageModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        layers = []
        layer_sizes = self.config.get("layer_sizes", [64, 64])
        assert "vocab_size" in self.config, "vocab_size must be specified in config"
        self.vocab_size = self.config["vocab_size"]
        # Add embedding layer if any.
        if self.config.get("has_embedding", False):
            embedding = nn.Embedding(self.vocab_size, layer_sizes[0])
            layers.append(embedding)
        # Add the last layer.
        for i in range(len(layer_sizes) - 1):
            self._add_layer(layers, layer_sizes[i], layer_sizes[i + 1])
        self._add_layer(layers, layer_sizes[-1], self.vocab_size, is_last_layer=True)
        self.ffm = nn.Sequential(*layers)

    def _add_layer(
        self,
        layers: List[nn.Module],
        in_size: int,
        out_size: int,
        is_last_layer: bool = False,
    ):
        layers.append(nn.Linear(in_size, out_size))
        if (
            not is_last_layer
            and self.config.get("activation_type", "relu").lower() == "relu"
        ):
            layers.append(nn.ReLU())

    def forward(self, batch) -> torch.Tensor:
        """Conduct a batch inference."""
        x = self.ffm(batch)
        return x
