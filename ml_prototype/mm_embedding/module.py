"""This file includes the modules used to construct the multi-modality item embedding learning model."""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageEncoder(nn.Module):
    """Configurable Image Encoder supporting different Hugging Face models.
    Args:
        model_name (str): Name of the Hugging Face vision model
            (e.g., 'facebook/resnet-50', 'google/vit-base-patch16-224-in21k').
        output_dim (int): Dimension of the final embedding.
        freeze (bool): Whether to freeze the encoder's parameters during training.
    """
    def __init__(self, model_name: str = 'microsoft/resnet-50', output_dim: int = 128, freeze: bool = True):
        super(ImageEncoder, self).__init__()
        self.model_name = model_name
        self.freeze = freeze
        # Load the pre-trained vision model
        self.model = AutoModel.from_pretrained(model_name)
        # Freeze the encoder if required
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        # Extract feature dimension from the model's configuration
        config = AutoConfig.from_pretrained(self.model_name)
        self.feature_dim = getattr(config, 'hidden_size', None) or getattr(config, 'embed_dim', None)

        # Manual fallback for known architectures like ResNet
        if self.feature_dim is None:
            if 'resnet' in model_name.lower():
                self.feature_dim = 2048  # ResNet-50 output size
            else:
                raise ValueError(f"Unknown feature dimension for model {model_name}")
        logging.debug(f"Feature dimension detected: {self.feature_dim}")

        # Fully connected layer to project features to the desired output dimension
        self.fc = nn.Linear(self.feature_dim, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Image Encoder.
        Args:
            images (torch.Tensor): Input images. Shape: (batch_size, 3, 224, 224)
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, output_dim).
        """
        # Extract features using the Hugging Face vision model
        outputs = self.model(pixel_values=images)
        # Extract the relevant features based on model output type
        if hasattr(outputs, 'last_hidden_state') and self.model_name.startswith('google/vit'):
            # For ViT models
            features = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        elif hasattr(outputs, 'pooler_output') and self.model_name.startswith('microsoft/resnet'):
            # For ResNet models, use pooler_output
            features = outputs.pooler_output  # Shape: (batch_size, hidden_size, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # Squeeze singleton dimensions
        elif isinstance(outputs, torch.Tensor):
            # Handle plain tensors from models like ResNet
            features = torch.mean(outputs, dim=[2, 3])  # Global Average Pooling
        else:
            raise ValueError(f"Model {self.model_name} does not have recognizable output features.")

        # Project to the desired output dimension
        embeddings = self.fc(features)
        return embeddings


class TextEncoder(nn.Module):
    """Text Encoder supporting multilingual BERT or XLM-RoBERTa."""
    def __init__(self, model_name='bert-base-multilingual-cased', device: str = "cpu", output_dim=512):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # Freeze the encoder
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(self.model.config.hidden_size, output_dim)

    def forward(self, texts: list[str]) -> torch.Tensor:
        # Tokenize and encode
        tokens = self.tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True, max_length=128
        )

        # Move tokens to the correct device
        tokens = {key: val.to(self.device) for key, val in tokens.items()}

        # Forward pass through the text encoder
        outputs = self.model(**tokens)
        # Extract the [CLS] token embedding, shape: (batch_size, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        # Project to output dimension
        x = self.fc(cls_embedding)  # shape: (batch_size, output_dim)
        return x


class FusionLayer(nn.Module):
    """Fusion Layer to combine image and text embeddings."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass of the fusion layer.
        It will accept the image and text embeddings and return the fused embedding.
        """
        combined = torch.cat((image_embedding, text_embedding), dim=1)
        output = self.activation(self.fc(combined))
        return output


class MultimodalEmbeddingModel(nn.Module):
    """Main model combining Image Encoder, Text Encoder, and Fusion Layer.
    It takes images and texts as input and returns the multimodal embeddings.
    """
    def __init__(self, image_encoder, text_encoder, fusion_layer, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.image_encoder = image_encoder.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.fusion_layer = fusion_layer.to(self.device)

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        image_embedding = self.image_encoder(images).to(self.device)
        text_embedding = self.text_encoder(texts).to(self.device)
        embedding = self.fusion_layer(image_embedding, text_embedding).to(self.device)
        return embedding
