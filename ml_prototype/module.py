"""Modules inherited from torch.nn.Module.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
