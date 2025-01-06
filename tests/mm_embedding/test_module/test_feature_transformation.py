"""This file contains the tests for the feature transformation functions.
Run this script with the command:
    pytest tests/mm_embedding/test_module/test_feature_transformation.py
"""
from ml_prototype.mm_embedding.module import IdentityTransformation, OneHotTransformation, EmbeddingTransformer
import torch
import pytest


class TestIdentityTransformation:
    def test_transform(self):
        transformer = IdentityTransformation(name="identity")
        input_data = torch.tensor([
            [1], [2], [5]
        ], dtype=torch.int32)
        output_data = transformer(input_data)
        assert torch.equal(output_data, input_data)


class TestOneHotTransformation:
    def test_transform(self):
        transformer = OneHotTransformation(name="one_hot", num_classes=4)
        input_data = torch.tensor([
            [1], [2], [3]
        ])
        output_data = transformer(input_data)
        expected_output = torch.tensor(
            [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32
        )
        assert torch.equal(output_data, expected_output), f"Expected: {expected_output} with shape {expected_output.shape}, got: {output_data} with shape {output_data.shape}"


class TestEmbeddingTransformer:
    @pytest.fixture(autouse=True)
    def setUp(self):
        """Set up test cases with predefined parameters."""
        self.name = "test_feature"
        self.num_embeddings = 100  # Number of unique categories
        self.embedding_dim = 16  # Dimension of embedding vector
        self.num_layers = 2  # Number of Transformer layers
        self.num_heads = 4  # Number of attention heads
        self.ff_dim = 32  # Feed-forward layer dimension
        self.dropout = 0.1  # Dropout rate
        self.batch_size = 8  # Number of samples in a batch
        self.num_oov_buckets = 1  # Out-of-vocabulary index

        # Initialize the EmbeddingTransformer
        self.transformer = EmbeddingTransformer(
            name=self.name,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            num_oov_buckets=self.num_oov_buckets
        )

        # Set the transformer to evaluation mode to disable dropout
        self.transformer.eval()

    def test_forward_pass(self):
        """Test the forward pass of EmbeddingTransformer."""
        # Create a batch of input tensors with shape [batch_size, 1]
        input_tensor = torch.randint(0, self.num_embeddings, (self.batch_size, 1))

        # Perform the forward pass
        output = self.transformer(input_tensor)

        # Check the output shape
        assert output.shape == (self.batch_size, self.embedding_dim), \
            f"Expected output shape to be ({self.batch_size}, {self.embedding_dim}), got {output.shape}"

    def test_deterministic_output(self):
        """Test if the output is deterministic given the same input."""
        input_tensor = torch.randint(0, self.num_embeddings, (self.batch_size, 1))

        # Perform the forward pass twice with the same input
        output1 = self.transformer(input_tensor)
        output2 = self.transformer(input_tensor)

        assert output1.shape == output2.shape, \
            f"Expected the output shapes to be identical, got {output1.shape} and {output2.shape}"

        # Check if the outputs are identical
        assert torch.allclose(output1, output2), \
            f"Expected the outputs to be identical, got {output1} and {output2}"
