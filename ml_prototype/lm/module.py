"""
"""
import abc
import math
from typing import Any, Dict, List

import torch
import torch.nn as nn


class LanguageModule(nn.Module, abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config


class RMSNorm(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        layer_size: int,
        eps: int = 1e-8,
        has_bias: bool = False,
    ):
        """Root mean square layer normalization.
        See Biao Zhang, Rico Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019.

        Args:
            config (Dict[str, Any]): The shared configuration for model.
            layer_size (int): The size of the layer,.
            eps (int, optional): The epsilon value. Defaults to 1e-8.
            has_bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.layer_size = layer_size
        self.eps = eps
        self.has_bias = has_bias
        self.scale = nn.Parameter(torch.ones(layer_size))
        self.register_parameter("scale", self.scale)
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(layer_size))
            self.register_parameter("bias", self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor.
        """
        # Only calculate the 2-norm on the last dimensions.
        norm_x = x.norm(p=2, dim=-1, keepdim=True)
        rms_x = self.layer_size**-0.5 * norm_x
        x_normed = x / (rms_x + self.eps)
        if self.has_bias:
            return self.scale * x_normed + self.bias
        else:
            return self.scale * x_normed


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: int = 1.0, dropout_ratio: int = 0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, q, k, v) -> torch.Tensor:
        """Forward pass for Scaled Dot Product Attention.
        An implementation of softmax(QK^T/sqrt(d_k)) * V.

        Args:
            q (torch.Tensor): The query tensor, in shape [n, q_len, d_k].
            k (torch.Tensor): The key tensor, in shape [n, k_len, d_k].
            v (torch.Tensor): The value tensor, in shape [n, k_len, d_v].

        Returns:
            torch.Tensor: The context vector after attention, in shape [n, k_len, d_v].
        """
        dk = q.shape[-1]
        attn = (q @ k) / math.sqrt(dk)
        attn = F.softmax(attn / self.temperature, dim=-1)
        output = torch.matmul(self.dropout(attn), v)
        return output


class Attention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_heads: int,
        temperature: int = 1.0,
        dropout_ratio: int = 0.0,
    ):
        """Constructor of the Attention class."""
        super(Attention, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Split embedding size into heads, validated by `embed_size % num_heads == 0`.
        assert (
            embed_size % num_heads == 0
        ), f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})."
        self.head_size = embed_size // num_heads

        # Initialize linear layers for projection and query.
        self.q = nn.Linear(self.hidden_size, self.embed_size)
        self.k = nn.Linear(self.hidden_size, self.embed_size)
        self.v = nn.Linear(self.hidden_size, self.embed_size)

        # Scaled dot product layer.
        self.scaled_dot_product_attention = ScaledDotProductAttention(
            temperature=temperature, dropout_ratio=dropout_ratio
        )

        # Output projection layer.
        self.output = nn.Linear(self.embed_size, self.hidden_size)

        # Allow registering additional attention layers.
        self.attention_layers = nn.ModuleDict()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Forward pass for the Attention module.

        Parameters:
        q (torch.Tensor): The query for attention, with shape (batch_size, hidden_size).
        k (torch.Tensor): The keys for attention, with shape (batch_size, seq_len, hidden_size).
        v (torch.Tensor): The values for attention, with shape (batch_size, seq_len, hidden_size).
        mask (torch.Tensor, optional): The attention mask, with shape (batch_size, seq_len). 1 indicates valid token, 0 indicates padding.

        Returns:
        torch.Tensor: The context vector after attention, with shape (batch_size, hidden_size).

        Functionality:
        1. Linearly project the keys, values and query using the module's linear layers.
        2. Split the projected keys, values and query into multiple heads.
        3. Apply scaled dot-product attention for each head.
        4. Concatenate the attention outputs of each head.
        5. Project the concatenated output through the output linear layer.
        """
        N = q.shape[0]

        q_len, k_len, v_len = q.shape[1], k.shape[1], v.shape[1]

        # Apply projection layer.
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # Reshape for multi-head attention.
        q = q.reshape(N, q_len, self.num_heads, self.head_size)
        k = k.reshape(N, k_len, self.num_heads, self.head_size)
        v = v.reshape(N, v_len, self.num_heads, self.head_size)

        # Compute scaled dot-product attention.
        attn = self.scaled_dot_product_attention(q, k, v)

        # Reshape for multi-head attention.
        attn = attn.reshape(N, q_len, self.num_heads * self.head_size)
        # Apply output projection layer.
        output = self.output(attn)
        return output


class FeedForwardModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        assert "embed_dim" in config, "FeedForwardModel requires an embed_dim."
        assert "vocab_size" in config, "FeedForwardModel requires a vocab_size."
        self.config = config
        embed_dim, vocab_size = config["embed_dim"], config["vocab_size"]
        layers = []
        assert (
            "ffm_layer_sizes" in config
        ), "ffm_layer_sizes must be specified in config"
        layer_sizes = config["ffm_layer_sizes"]
        self._add_layer(layers, embed_dim, layer_sizes[0])
        for i in range(len(layer_sizes) - 1):
            self._add_layer(layers, layer_sizes[i], layer_sizes[i + 1])
        # Add the last layer.
        if config.get("add_last_layer", False):
            self._add_layer(layers, layer_sizes[-1], vocab_size, is_last_layer=True)
        self.ffm = nn.Sequential(*layers)

    def _add_layer(
        self,
        layers: List[nn.Module],
        in_size: int,
        out_size: int,
        is_last_layer: bool = False,
    ):
        if self.config.get("pre_norm", False):
            norm_type = self.config.get("norm_type", "layer_norm")
            if norm_type == "layer_norm":
                layers.append(nn.LayerNorm(in_size))
            elif norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(in_size))
            elif norm_type == "rms_norm":
                layers.append(RMSNorm(self.config, in_size))
        layers.append(nn.Linear(in_size, out_size))
        if (
            not is_last_layer
            and self.config.get("activation_type", "relu").lower() == "relu"
        ):
            layers.append(nn.ReLU())
        if not self.config.get("pre_norm", False):
            norm_type = self.config.get("norm_type", "layer_norm")
            if norm_type == "layer_norm":
                layers.append(nn.LayerNorm(out_size))
            elif norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(out_size))
            elif norm_type == "rms_norm":
                layers.append(RMSNorm(self.config, out_size))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Conduct a batch inference.
        The batch is expected to be in shape of [batch_size, seq_len, input_dim].
        """
        x = self.ffm(batch)
        return x


class FeedForwardLM(LanguageModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert "embed_dim" in config, "embed_dim must be specified in config"
        embed_dim = config["embed_dim"]
        assert "vocab_size" in config, "vocab_size must be specified in config"
        vocab_size = config["vocab_size"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.feed_forward = FeedForwardModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.feed_forward(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embed_dim = config.get("embed_dim", 256)
        self.num_heads = config.get("num_heads", 4)
        self.dropout_ratio = config.get("dropout_ratio", 0.0)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_ratio,
            batch_first=True,
        )
        config.update({"ffm_layer_sizes": [self.embed_dim, self.embed_dim]})
        self.feed_forward = FeedForwardModel(config)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): The input tensor. Shape: [batch_size, seq_len, embed_dim].
        """
        # assert (
        #     x.shape[-1] == self.embed_dim
        # ), f"x.shape: {x.shape}, embed_dim: {self.embed_dim}"
        # Attention with residual.
        x = self.norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)
        # Feed forward with residual.
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        return x


class StackedTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_layers = config.get("num_layers", 1)
        self.embed_dim = config.get("embed_dim", 256)
        # List to hold the transformer blocks
        layers = []
        for _ in range(self.num_layers):
            layers.append(TransformerBlock(config))
        self.stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Stacked Transformer.
        Args:
            x (torch.Tensor): The input tensor. Shape: [batch_size, seq_len, embed_dim].
        """
        # Pass the input through the stacked transformer blocks
        x = self.stack(x)
        return x


class SinCosPositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pos_emb = torch.zeros(seq_len, embed_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_emb[: x.size(1), :]


class TransformerLM(LanguageModule):
    """TransformerLM is a language model that leverages the transformer architecture.
    It consists of an embedding layer followed by a stack of transformer blocks.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing parameters for the model.
            - "input_dim": The dimension of the input embeddings.
            - "vocab_size": The size of the vocabulary.
            - Other parameters required by the StackedTransformer class.

    Example usage:
        config = {
            "vocab_size": 10000,
            "num_layers": 4,
            "embed_dim": 256,
            "num_heads": 8,
            "dropout": 0.1
        }
        model = TransformerLM(config)
        x = torch.Tensor(batch_size, seq_len).long()
        output = model(x)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        assert "embed_dim" in config, "embed_dim must be specified in config"
        embed_dim = config["embed_dim"]
        assert "vocab_size" in config, "vocab_size must be specified in config"
        vocab_size = config["vocab_size"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = SinCosPositionalEmbedding(
            seq_len=config["context_size"], embed_dim=config["embed_dim"]
        )
        self.stacked_transformer = StackedTransformer(config)
        self.projection_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TransformerLM.

        Args:
            x (torch.Tensor): The input tensor containing token IDs.
                Shape: [batch_size, seq_len, embed_dim].

        Returns:
            torch.Tensor: The output tensor after passing through the embedding and transformer layers.
                Shape: [batch_size, seq_len, vocab_size].
        """
        x = self.embedding(x)
        x = self.position_embedding(x)
        x = self.stacked_transformer(x)
        logits = self.projection_layer(x)
        return logits
