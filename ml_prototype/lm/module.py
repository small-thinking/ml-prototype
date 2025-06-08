"""
"""
import abc
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler


class LanguageModule(nn.Module, abc.ABC):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int):
        super(SwiGLU, self).__init__()
        self.embed_dim = embed_dim

        # Gated layer weights and bias
        self.gate_weight = nn.Parameter(torch.Tensor(embed_dim))
        self.gate_bias = nn.Parameter(torch.Tensor(embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gate_weight, -0.1, 0.1)
        nn.init.zeros_(self.gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish Part
        swish = x * torch.sigmoid(x)

        # Reshape the 1D gate_weight and gate_bias to 3D for linear transformation
        gate_weight_3D = self.gate_weight[None, None, :]
        gate_bias_3D = self.gate_bias[None, None, :]

        # Gated Linear Unit Part
        # Use broadcasting to apply the gate across the batch and sequence length
        gate = torch.sigmoid((x * gate_weight_3D) + gate_bias_3D)

        # SwiGLU
        out = swish * gate

        return out


class RMSNorm(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        layer_shape: Union[int, Tuple[int, int]],
        eps: float = 1e-8,
        has_bias: bool = False,
    ):
        """Root mean square layer normalization.
        See Biao Zhang, Rico Sennrich, Root Mean Square Layer Normalization, NeurIPS 2019.

        Args:
            config (Dict[str, Any]): The shared configuration for the model.
            layer_shape (Union[int, Tuple[int, int]]): Either the embedding dimension or a
                tuple ``(seq_len, embed_dim)``.
            eps (float, optional): The epsilon value. Defaults to 1e-8.
            has_bias (bool, optional): Whether to use bias. Defaults to False.
        """
        super().__init__()
        self.config = config
        self.layer_shape = layer_shape
        self.eps = eps
        self.has_bias = has_bias

        if isinstance(layer_shape, tuple):
            embed_dim = layer_shape[-1]
        else:
            embed_dim = layer_shape

        self.scale = nn.Parameter(torch.ones(embed_dim))

        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.bias = torch.zeros(embed_dim, requires_grad=False)

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
        x_normed = x / rms_x

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


class CustomScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_ratio: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_ratio: float,
        attn_mask: torch.Tensor = None,
        is_causal: bool = False,  # Useless parameter for interface compatibility
    ) -> torch.Tensor:
        """Forward pass for Scaled Dot Product Attention.

        Args:
            q (torch.Tensor): The query tensor, in shape [batch_size, num_heads, seq_len, head_dim]
            k (torch.Tensor): The key tensor, in shape [batch_size, num_heads, seq_len, head_dim]
            v (torch.Tensor): The value tensor, in shape [batch_size, num_heads, seq_len, head_dim]
            dropout_ratio (float): Useless parameter for interface compatibility.
            attn_mask (torch.Tensor): The attention mask to apply to the input.
            is_causal (bool): Useless parameter for interface compatibility.

        Returns:
            torch.Tensor: The context vector after attention, in shape [batch_size, seq_len, embed_dim, num_heads].
        """
        dk = q.shape[-1]  # head_dim
        # Compute the dot product between query and key tensors.
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(dk)  # Shape: [batch_size, num_heads, seq_len, seq_len]
        assert attn.shape == (q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        assert attn_mask is not None
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1)  # Shape: [batch_size, num_heads, seq_len, seq_len]
        attn = self.dropout(attn)
        # Compute the context vector by taking a weighted sum of the values.
        output = attn @ v  # Shape: [batch_size, num_heads, seq_len, head_dim]
        assert output.shape == (q.shape[0], q.shape[1], q.shape[2], v.shape[3])
        return output


class CustomAttention(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Constructor of the Attention class."""
        super().__init__()

        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.dropout_ratio = config.get("dropout_ratio", 0.01)

        # Split embedding size into heads, validated by `embed_size % num_heads == 0`.
        assert (
            self.embed_dim % self.num_heads == 0
        ), f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})."

        self.head_dim = self.embed_dim // self.num_heads

        # Initialize linear layers for query, key, and value projections.
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

        # Scaled dot product layer.
        if not config.get("use_customized_scaled_doc_product_attention", False):
            self.scaled_dot_product_attention = (
                F.scaled_dot_product_attention
            )  # torch.nn.scaled_dot_product_attention
        else:
            self.scaled_dot_product_attention = CustomScaledDotProductAttention(
                dropout_ratio=self.dropout_ratio
            )

        # Output projection layer.
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

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
        )  # Shape: [batch_size, num_heads, seq_len, head_dim]
        K = K.reshape(batch_size, k_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # Shape: [batch_size, num_heads, seq_len, head_dim]
        V = V.reshape(batch_size, v_len, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # Shape: [batch_size, num_heads, seq_len, head_dim]

        # Step 3: Compute scaled dot-product attention.
        attn_output = self.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=attn_mask,
            dropout_ratio=self.dropout_ratio,
            is_causal=is_causal,
        )  # Shape: [batch_size, num_heads, seq_len, head_dim]

        # Step 4 and 5: Reshape and project output.
        attn_output = (
            attn_output.permute(0, 2, 1, 3)  # Shape after permute: [batch_size, seq_len, num_heads, head_dim]
            .contiguous()
            .view(batch_size, q_len, self.embed_dim)
        )  # Shape: [batch_size, seq_len, embed_dim = num_heads * head_dim]
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
        if not is_last_layer:
            if self.config.get("activation_type", "relu").lower() == "relu":
                layers.append(nn.ReLU())
            elif self.config.get("activation_type", "gelu").lower() == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(SwiGLU(out_size))
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
        activate_type = config.get("activation_type", "relu").lower()
        if activate_type == "gelu":
            activation = nn.GELU()
        elif activate_type == "swiglu":
            activation = SwiGLU(4 * embed_dim)
        else:
            activation = nn.ReLU()

        # Feed forward layer is 4x wider of the embedding dimension
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            activation,
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

        if not config.get("use_custom_attention", False):
            self.attention = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.dropout_ratio,
                batch_first=True,
            )
            self.is_causal = True
        else:
            self.attention = CustomAttention(config=config)
            self.is_causal = False
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
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
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
        dropout_rate = config.get("dropout_ratio", 0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual_seq_len = min(x.shape[1], self.seq_len)
        pos_embedding = self.pos_embedding_table(
            torch.arange(actual_seq_len, device=x.device)
        )
        x = x + pos_embedding
        x = self.dropout(x)
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


class WarmupCosineDecayScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 2000,
        target_lr: float = 0.0003,
        min_lr: float = 0.00001,
        steps_in_cycle: int = 1000000,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        """
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to which the learning rate scheduler will be attached.
            warmup_steps (int): Number of steps over which to increase the learning rate from 0 to target_lr.
            target_lr (float): Learning rate to reach at the end of the warmup phase.
            min_lr (float): Minimum learning rate during the cosine decay phase.
            steps_in_cycle (int): Total number of steps in a cosine decay cycle.
            last_epoch (int, optional): The index of the last epoch. Default is -1.
            verbose (bool): Whether to print the lr.
        """
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.min_lr = min_lr
        self.steps_in_cycle = steps_in_cycle
        super(WarmupCosineDecayScheduler, self).__init__(
            optimizer, last_epoch, verbose=verbose
        )

    def get_lr(self) -> List[float]:
        if self._step_count < self.warmup_steps:
            # Warmup phase.
            alpha = self._step_count / self.warmup_steps
            lr_val = self.target_lr * alpha
        elif self._step_count < self.steps_in_cycle:
            # Cosine decay stage.
            progress = (self._step_count - self.warmup_steps) / (
                self.steps_in_cycle - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr_val = self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay
        else:
            lr_val = self.min_lr
        return [lr_val for _ in self.base_lrs]


