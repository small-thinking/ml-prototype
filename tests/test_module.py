"""Unit tests for the implemented modules.
"""
import pytest
import torch

from ml_prototype.lm.module import MultiHeadAttention, RMSNorm


@pytest.mark.parametrize(
    "input_shape, layer_size, eps, has_bias",
    [
        ((2, 3, 4), 4, 1e-8, False),
        ((2, 3, 4), 4, 1e-8, True),
        ((5, 5, 5), 5, 1e-6, False),
        ((5, 5, 5), 5, 1e-6, True),
    ],
)
def test_rmsnorm(input_shape, layer_size, eps, has_bias):
    config = {
        "context_size": input_shape[1]
    }  # Add any additional config parameters here
    x = torch.rand(input_shape)
    rmsnorm = RMSNorm(config, layer_size, eps, has_bias)
    output = rmsnorm(x)

    # Check the output shape
    assert output.shape == input_shape

    # Check that the output is not all zeros
    assert torch.any(output != 0)


class TestMultiHeadAttention:
    def test_initialization(self):
        config = {
            "input_dim": 16,
            "num_heads": 8,
            "dim_model": 256,
        }
        mha = MultiHeadAttention(config)
        assert mha.input_dim == 16
        assert mha.num_heads == 8
        assert mha.dim_model == 256
        assert mha.dim_head == 32

    @pytest.mark.parametrize(
        "batch_size, context_size, input_dim, dim_model, num_heads",
        [(64, 50, 16, 256, 8), (32, 100, 16, 512, 4)],
    )
    def test_split_and_combine_heads(
        self, batch_size, context_size, input_dim, dim_model, num_heads
    ):
        config = {
            "input_dim": input_dim,
            "num_heads": num_heads,
            "dim_model": dim_model,
        }
        mha = MultiHeadAttention(config)
        x = torch.rand(batch_size, context_size, dim_model)
        split_heads = mha._split_heads(x)
        assert split_heads.shape == (
            batch_size,
            num_heads,
            context_size,
            dim_model // num_heads,
        )
        combined_heads = mha._combine_heads(split_heads)
        assert combined_heads.shape == (batch_size, context_size, dim_model)

    @pytest.mark.parametrize(
        "batch_size, context_size, input_dim, dim_model, num_heads",
        [(64, 50, 16, 256, 8), (32, 100, 16, 512, 4)],
    )
    def test_scaled_dot_product_attention(
        self, batch_size, context_size, input_dim, dim_model, num_heads
    ):
        config = {
            "input_dim": input_dim,
            "num_heads": num_heads,
            "dim_model": dim_model,
        }
        mha = MultiHeadAttention(config)
        Q = torch.rand(batch_size, num_heads, context_size, dim_model // num_heads)
        K = torch.rand(batch_size, num_heads, context_size, dim_model // num_heads)
        V = torch.rand(batch_size, num_heads, context_size, dim_model // num_heads)
        attn_output = mha._scaled_dot_product_attention(Q, K, V)
        assert attn_output.shape == (
            batch_size,
            num_heads,
            context_size,
            dim_model // num_heads,
        )

    @pytest.mark.parametrize(
        "batch_size, context_size, input_dim, dim_model, num_heads",
        [(64, 50, 16, 256, 8), (32, 100, 16, 512, 4)],
    )
    def test_forward(self, batch_size, context_size, input_dim, dim_model, num_heads):
        config = {
            "input_dim": input_dim,
            "num_heads": num_heads,
            "dim_model": dim_model,
        }
        mha = MultiHeadAttention(config)
        Q = torch.rand(batch_size, context_size, input_dim)
        K = torch.rand(batch_size, context_size, input_dim)
        V = torch.rand(batch_size, context_size, input_dim)
        output = mha.forward(Q, K, V)
        assert output.shape == (batch_size, context_size, dim_model)
