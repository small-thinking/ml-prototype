"""
"""
import abc
import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiHeadAttention(nn.Module):
    """A multi head attention."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        # Validate parameters.
        self.config = config
        if "input_dim" not in config:
            raise ValueError("input_dim must be specified in config")
        self.input_dim = config["input_dim"]
        self.num_heads = config.get("num_heads", 4)
        self.dim_model = config.get("dim_model", self.num_heads * 128)
        assert (
            self.dim_model % self.num_heads == 0
        ), f"dim_model must be multiplier of num_heads: dim_model: {self.dim_model}, num_heads: {self.num_heads}"
        self.dim_head = self.dim_model // self.num_heads
        # Weights for self-attention.
        self.W_Q = nn.Linear(self.input_dim, self.dim_model)
        self.W_K = nn.Linear(self.input_dim, self.dim_model)
        self.W_V = nn.Linear(self.input_dim, self.dim_model)
        self.W_O = nn.Linear(self.dim_model, self.dim_model)

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """Apply scaled dot product attention. The input Q, K, V are already splitted into heads.
        Args:
            Q (torch.Tensor): The query tensor. Expected dimension of (batch_size, num_heads, context_size, dim_head).
            K (torch.Tensor): The key tensor. Expected dimension of (batch_size, num_heads, context_size, dim_head).
            V (torch.Tensor): The value tensor. Expected dimension of (batch_size, num_heads, context_size, dim_head).
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.

        Return:
            torch.Tensor: The output tensor. Expected dimension of (batch_size, num_heads, context_size, dim_head).
        """
        # Calculate the scaled dot product attention. attn_scores = QK^T / sqrt(dim_head).
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dim_head)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # Calculate the softmax. formula: softmax(attn_scores).
        attn_probs = F.softmax(attn_scores, dim=-1)
        # Calculate the attention output. formula: V * softmax(attn_scores) * Q.
        attn_output = torch.matmul(attn_probs, V)
        return attn_output

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Not really split, but a reshape of the matrix into sub-space.

        Args:
            x (torch.Tensor): The input tensor, with expected dimenions of (batch_size, context_size, dim_model).
        """
        batch_size, context_size, _ = x.shape
        # After reshaping, the dimension is expected to be (batch_size, num_heads, context_size, dim_head).
        return x.reshape(
            batch_size, context_size, self.num_heads, self.dim_head
        ).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the sub-space back to the matrix.

        Args:
            x (torch.Tensor): The input tensor, expected dimenions of (batch_size, num_heads, context_size, dim_head).
        """
        batch_size, num_heads, context_size, dim_head = x.shape
        return x.transpose(1, 2).reshape(batch_size, context_size, num_heads * dim_head)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Perform multi-head attention. Q, K, V, are expected to be the same.
        Args:
            Q (torch.Tensor): The query tensor. Expected dimension of (batch_size, context_size, input_dim).
            K (torch.Tensor): The key tensor. Expected dimension of (batch_size, context_size, input_dim).
            V (torch.Tensor): The value tensor. Expected dimension of (batch_size, context_size, input_dim).
            mask (torch.Tensor, optional): The mask tensor. Defaults to None.
        """
        # Linear layer for self-attention.
        Q = self._split_heads(self.W_Q(Q))
        K = self._split_heads(self.W_K(K))
        V = self._split_heads(self.W_V(V))
        # Calculate the attention output.
        attn_output = self._scaled_dot_product_attention(Q, K, V, mask)
        # Combine the sub-space back to the matrix.
        output = self.W_O(self._combine_heads(attn_output))
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: int = 1.0, dropout_ratio: int = 0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for Scaled Dot Product Attention.
        An implementation of softmax(QK^T/sqrt(d_k)) * V.

        Args:
            Q (torch.Tensor): The query tensor, in shape [n, q_len, num_heads, d_k].
            K (torch.Tensor): The key tensor, in shape [n, k_len, num_heads, d_k].
            V (torch.Tensor): The value tensor, in shape [n, v_len, num_heads, d_v].
            mask (torch.Tensor, optional): The mask tensor. Defaults to None. Expected shape [n, q_len, k_len].

        Returns:
            torch.Tensor: The context vector after attention, in shape [n, q_len, num_heads, d_v].
        """
        d_k = Q.shape[-1]
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # Shape: [n, q_len, num_heads, k_len]
        if mask is not None:
            attn = attn.masked_fill(
                mask == 0, -1e9
            )  # Shape: [n, q_len, num_heads, k_len]
        attn = F.softmax(
            attn / self.temperature, dim=-1
        )  # Shape: [n, q_len, num_heads, k_len]
        output = torch.matmul(
            self.dropout(attn), V
        )  # Shape: [n, q_len, num_heads, d_v]
        return output


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        num_heads: int,
        temperature: int = 1.0,
        dropout_ratio: int = 0.0,
    ):
        """Constructor of the Attention class."""
        super(MultiHeadAttentionLayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # Split embedding size into heads, validated by `embed_size % num_heads == 0`.
        assert (
            embed_size % num_heads == 0
        ), f"embed_size ({embed_size}) must be divisible by num_heads ({num_heads})."
        self.head_size = embed_size // num_heads
        # Initialize linear layers for projection and query.
        self.W_Q = nn.Linear(self.hidden_size, self.embed_size)
        self.W_K = nn.Linear(self.hidden_size, self.embed_size)
        self.W_V = nn.Linear(self.hidden_size, self.embed_size)
        # Scaled dot product layer.
        self.scaled_dot_product_attention = ScaledDotProductAttention(
            temperature=temperature, dropout_ratio=dropout_ratio
        )
        # Output projection layer.
        self.output = nn.Linear(self.embed_size, self.hidden_size)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Forward pass for the Attention module. For self-attention, the query, key and value are all the same.

        Parameters:
        Q (torch.Tensor): The query for attention, with shape (batch_size, seq_len, hidden_size).
        K (torch.Tensor): The keys for attention, with shape (batch_size, seq_len, hidden_size).
        V (torch.Tensor): The values for attention, with shape (batch_size, seq_len, hidden_size).
        mask (torch.Tensor, optional): The attention mask, with shape (batch_size, seq_len).
            Defaults to None, or 1 indicates valid token, 0 indicates padding.

        Returns:
        torch.Tensor: The context vector after attention, with shape (batch_size, hidden_size).

        Functionality:
        1. Linearly project the keys, values and query using the module's linear layers.
        2. Split the projected keys, values and query into multiple heads.
        3. Apply scaled dot-product attention for each head.
        4. Concatenate the attention outputs of each head.
        5. Project the concatenated output through the output linear layer.
        """
        N = Q.shape[0]
        q_len, k_len, v_len = Q.shape[1], K.shape[1], V.shape[1]

        # Apply projection layer.
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # Reshape for multi-head attention.
        Q = Q.reshape(N, q_len, self.num_heads, self.head_size)
        K = K.reshape(N, k_len, self.num_heads, self.head_size)
        V = V.reshape(N, v_len, self.num_heads, self.head_size)

        # Compute scaled dot-product attention.
        attn = self.scaled_dot_product_attention(Q, K, V)

        # Reshape for multi-head attention.
        attn = attn.reshape(N, q_len, self.num_heads * self.head_size)
        # Apply output projection layer.
        output = self.output(attn)
        return output


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
        for i in range(len(layer_sizes) - 1):
            self._add_layer(layers, layer_sizes[i], layer_sizes[i + 1])
        # Add the last layer.
        self._add_layer(layers, layer_sizes[-1], self.vocab_size, is_last_layer=True)
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

    def forward(self, batch) -> torch.Tensor:
        """Conduct a batch inference."""
        x = self.ffm(batch)
        return x
