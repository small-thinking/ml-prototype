"""
"""
import abc
import math
from typing import Any, Dict, List, Optional, Tuple

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
        layer_shape: Tuple[int, int],
        eps: float = 1e-8,
        has_bias: bool = False,
    ):
        """Root mean square layer normalization.
        See Biao Zhang, Rico Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019.

        Args:
            config (Dict[str, Any]): The shared configuration for the model.
            layer_shape (Tuple[int, int]): The shape of the layer [seq_len, embed_dim].
            eps (float, optional): The epsilon value. Defaults to 1e-8.
            has_bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.layer_shape = layer_shape  # Shape: [seq_len, embed_dim]
        self.eps = eps
        self.has_bias = has_bias
        self.scale = nn.Parameter(torch.ones(layer_shape[1]))  # Shape: [embed_dim]

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(layer_shape[1]))  # Shape: [embed_dim]
        else:
            self.bias = torch.zeros(
                layer_shape[1], requires_grad=False
            )  # Shape: [embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor with shape [batch, seq_len, embed_dim]
        """

        # Calculate RMS value along the last dimension (embed_dim)
        # Shape of rms_x: [batch, seq_len, 1]
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        # Normalize x by RMS value
        # Shape of x_normed: [batch, seq_len, embed_dim]
        x_normed = x / (rms_x + self.eps)

        # Scale the normalized output
        # Shape of self.scale.unsqueeze(0).unsqueeze(0): [1, 1, embed_dim]
        # Broadcasting takes care of the rest
        # Final shape: [batch, seq_len, embed_dim]
        if self.has_bias:
            return self.scale.unsqueeze(0).unsqueeze(
                0
            ) * x_normed + self.bias.unsqueeze(0).unsqueeze(0)
        else:
            return self.scale.unsqueeze(0).unsqueeze(0) * x_normed


# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, temperature: int = 1.0, dropout_ratio: int = 0.0):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_ratio)

#     def forward(self, q, k, v) -> torch.Tensor:
#         """Forward pass for Scaled Dot Product Attention.
#         An implementation of softmax(QK^T/sqrt(d_k)) * V.

#         Args:
#             q (torch.Tensor): The query tensor, in shape [n, q_len, d_k].
#             k (torch.Tensor): The key tensor, in shape [n, k_len, d_k].
#             v (torch.Tensor): The value tensor, in shape [n, k_len, d_v].

#         Returns:
#             torch.Tensor: The context vector after attention, in shape [n, k_len, d_v].
#         """
#         dk = q.shape[-1]
#         attn = (q @ k) / math.sqrt(dk)
#         attn = F.softmax(attn, dim=-1)
#         output = torch.matmul(self.dropout(attn), v)
#         return output


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_ratio: int = 0.0,
    ):
        """Constructor of the Attention class."""
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_ratio = dropout_ratio

        # Split embedding size into heads, validated by `embed_size % num_heads == 0`.
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."

        self.head_dim = embed_dim // num_heads

        # Initialize linear layers for query, key, and value projections.
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Scaled dot product layer.
        self.scaled_dot_product_attention = (
            F.scaled_dot_product_attention
        )  # torch.nn.scaled_dot_product_attention

        # Output projection layer.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Attention module.

        Parameters:
            query (torch.Tensor): The query for attention, shape (batch_size, seq_len, embed_dim).
            key (torch.Tensor): The keys for attention, shape (batch_size, seq_len, embed_dim).
            value (torch.Tensor): The values for attention, shape (batch_size, seq_len, embed_dim).
            attn_mask (torch.Tensor, optional): The attention mask, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output after attention, shape (batch_size, seq_len, embed_dim).
            Optional[Tensor]: Placeholder.

        Functionality:
            1. Linearly project the keys, values, and query.
            2. Split the projected keys, values, and query into multiple heads.
            3. Apply scaled dot-product attention.
            4. Concatenate the attention outputs.
            5. Project the concatenated output.
        """
        batch_size = query.size(0)
        q_len, k_len, v_len = query.size(1), key.size(1), value.size(1)

        # Step 1: Apply projection layer.
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Step 2: Reshape for multi-head attention.
        Q = Q.reshape(batch_size, q_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        K = K.reshape(batch_size, k_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        V = V.reshape(batch_size, v_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

        # Step 3: Compute scaled dot-product attention.
        attn_output = self.scaled_dot_product_attention(
            Q, K, V, attn_mask, self.dropout_ratio, is_causal
        )

        # Step 4 and 5: Reshape and project output.
        attn_output = (
            attn_output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, q_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        return output, None


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


class FeedForward(nn.Module):
    """a simple feed forward layer followed by a non-linearity"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        assert "embed_dim" in config, "FeedForwardModel requires an embed_dim."
        self.config = config
        embed_dim = config["embed_dim"]
        dropout = config.get("dropout_ratio", 0.01)
        # Feed forward layer is 4x wider of the embedding dimension
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.dropout_ratio = config.get("dropout_ratio", 0.0)
        self.seq_len = config["seq_len"]
        self.norm_type = config.get("norm_type", "simple")

        if config.get("use_custom_attention", False):
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_ratio,
                batch_first=True,
            )
        else:
            self.attention = Attention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout_ratio=self.dropout_ratio,
            )
        self.feed_forward = FeedForward(config)

        if self.norm_type == "rms":
            self.norm1 = RMSNorm(config, (self.seq_len, self.embed_dim))
            self.norm2 = RMSNorm(config, (self.seq_len, self.embed_dim))
        else:
            self.norm1 = nn.LayerNorm(self.embed_dim)
            self.norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): The input tensor. Shape: [batch_size, seq_len, embed_dim].
        """
        assert (
            x.shape[-1] == self.embed_dim
        ), f"x.shape: {x.shape}, embed_dim: {self.embed_dim}"
        # Attention with residual.
        norm_x = self.norm1(x)
        attn_output, _ = self.attention(
            query=norm_x, key=norm_x, value=norm_x, attn_mask=None, is_causal=True
        )
        x = x + attn_output
        # Feed forward with residual.
        x = x + self.feed_forward(self.norm2(x))
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

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the Stacked Transformer.
        Args:
            x (torch.Tensor): The input tensor. Shape: [batch_size, seq_len, embed_dim].
        """
        for layer in self.stack:
            x = layer(x, attn_mask=attn_mask)
        return x


class SimplePositionEmbedding(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.seq_len, self.embed_dim = config["seq_len"], config["embed_dim"]
        self.pos_embedding_table = nn.Embedding(self.seq_len, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_seq_len = min(x.shape[1], self.seq_len)
        pos_embedding = self.pos_embedding_table(
            torch.arange(actual_seq_len, device=x.device)
        )
        x = x + pos_embedding
        return x


class SinCosPositionalEmbedding(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.seq_len, self.embed_dim = (
            config["seq_len"],
            config["embed_dim"],
        )
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(
            1
        )  # [seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float()
            * (-math.log(10000.0) / self.embed_dim)
        )
        pos_emb = torch.zeros(self.seq_len, self.embed_dim)
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_seq_len = min(x.shape[1], self.seq_len)
        x = x * math.sqrt(self.embed_dim) + self.pos_emb[:actual_seq_len, :]
        return x


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

    def __init__(self, config: Dict[str, Any], device):
        super().__init__(config)
        assert "embed_dim" in config, "embed_dim must be specified in config"
        self.embed_dim = config["embed_dim"]
        assert "vocab_size" in config, "vocab_size must be specified in config"
        vocab_size = config["vocab_size"]
        self.use_position_embedding = config.get("use_position_embedding", True)
        self.pos_embedding_type = config.get("pos_embedding_type", "simple")

        # Embedding for tokens
        self.embedding = nn.Embedding(vocab_size, self.embed_dim)
        if self.pos_embedding_type == "none":
            self.pos_embedding = torch.Tensor.zeros(
                config["seq_len"], self.embed_dim, requires_grad=False
            )
        elif self.pos_embedding_type == "simple":
            self.pos_embedding = SimplePositionEmbedding(config)
        else:
            self.pos_embedding = SinCosPositionalEmbedding(config)

        self.stacked_transformer = StackedTransformer(config)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.projection_layer = nn.Linear(self.embed_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape
        # # Assertions to check the attention mask
        if attn_mask is not None:
            assert (
                attn_mask.shape[-1] == attn_mask.shape[-2] == seq_len
            ), f"Attention mask last two dimensions should be of size {seq_len}"

            # Check for causal mask
            if torch.any(attn_mask == 1):
                assert torch.all(
                    attn_mask.tril() == attn_mask
                ), "Attention mask is not an upper triangular matrix, thus not causal."

        # Token embedding
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        assert x.shape == (
            batch_size,
            seq_len,
            self.embed_dim,
        ), f"x.shape = {x.shape}, expected shape ({batch_size}, {seq_len}, {self.embed_dim})"

        # Apply pos embedding
        x = self.pos_embedding(x)

        # Passing through stacked transformers with attention mask
        x = self.stacked_transformer(x, attn_mask=attn_mask)
        x = self.layer_norm(x)
        logits = self.projection_layer(x)

        return logits
