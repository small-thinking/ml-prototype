"""Unit tests for the implemented modules.
"""
import pytest
import torch

from ml_prototype.lm.module import RMSNorm


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
