"""This file includes the modules used to construct the multi-modality item embedding learning model.
Run the script with the command:
    python -m ml_prototype.mm_embedding.module --config ml_prototype/mm_embedding/config.yaml
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging
from ml_prototype.mm_embedding.util import Logger
from ml_prototype.mm_embedding.config import FeatureTransformationConfig, MyConfig
from jsonargparse import ArgumentParser, ActionConfigFile
import importlib


logging.basicConfig(level=logging.DEBUG)


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


class BaseFeatureTransformation(nn.Module):
    """
    Base class for feature transformations.
    Derived classes should implement the forward method.
    """
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def output_dim(self) -> int:
        raise NotImplementedError("Derived classes must implement the output_dim method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Derived classes must implement the forward method.")


class IdentityTransformation(BaseFeatureTransformation):
    """
    Identity transformation module that applies f(x) = x.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def output_dim(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FeatureScaling(BaseFeatureTransformation):
    """
    Feature scaling module that scales features to a specified range using min-max normalization.
    """
    def __init__(self, name: str, min_val: float, max_val: float):
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val

    def output_dim(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / (self.max_val - self.min_val)


class OneHotTransformation(BaseFeatureTransformation):
    """
    One-hot transformation module that converts categorical features to one-hot encoded tensors.
    """
    def __init__(self, name: str, num_classes: int):
        super().__init__(name)
        self.num_classes = num_classes

    def output_dim(self) -> int:
        return self.num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Converts a tensor of indices to one-hot encoded tensor.

        Args:
            x (torch.Tensor): Input tensor with shape [b, 1].

        Returns:
            torch.Tensor: One-hot encoded tensor with shape [b, num_classes].
        """
        # tensor has shape [b, 1] with the index of the category, convert to one-hot with shape [b, num_classes]
        x = x.long()
        return F.one_hot(x, num_classes=self.num_classes).squeeze(1).float()


class EmbeddingTransformer(BaseFeatureTransformation):
    """
    Composite transformation: Embedding lookup + N layers of Transformer.
    """
    def __init__(
        self,
        name: str,
        num_embeddings: int,
        embedding_dim: int = 8,
        num_layers: int = 1,
        num_heads: int = 4,
        ff_dim: int = 16,
        dropout: float = 0.1,
        num_oov_buckets: int = 1
    ):
        """
        Args:
            name (str): Name of the feature.
            num_embeddings (int): Number of unique categories.
            embedding_dim (int): Dimension of the embedding vector.
            num_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads in the Transformer.
            ff_dim (int): Dimension of the feed-forward layer in the Transformer.
            dropout (float): Dropout rate for Transformer layers.
            num_oov_buckets (int): Number of OOV buckets for hashing OOV indices. Defaults to 1.
        """
        super().__init__(name)
        self.num_embeddings = num_embeddings
        self.num_oov_buckets = num_oov_buckets
        self.embedding = nn.Embedding(num_embeddings=num_embeddings + num_oov_buckets, embedding_dim=embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def output_dim(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, 1].

        Returns:
            torch.Tensor: Transformed tensor with shape [batch_size, embedding_dim].
        """
        # Handle out-of-vocabulary (OOV) indices
        x = x.long()
        oov_mask = x >= self.num_embeddings
        x[oov_mask] = self.num_embeddings + (x[oov_mask] % self.num_oov_buckets)

        # Perform embedding lookup
        embedding = self.embedding(x).squeeze(1)  # Shape: [batch_size, embedding_dim]

        # Add a sequence dimension for Transformer input (required shape: [seq_len, batch_size, embedding_dim])
        embedding = embedding.unsqueeze(0)  # Shape: [1, batch_size, embedding_dim]

        # Apply the Transformer
        transformed = self.transformer(embedding)  # Shape: [1, batch_size, embedding_dim]

        # Remove the sequence dimension and return
        return transformed.squeeze(0)  # Shape: [batch_size, embedding_dim]


class TabularEncoder(nn.Module):
    """
    Tabular Encoder module that applies transformations to tabular features.
    """
    def __init__(self, feature_transformation_configs: dict[str, FeatureTransformationConfig]):
        super().__init__()
        self.logger = Logger(__name__)
        self.feature_transformations = nn.ModuleDict()
        for name, transformation in feature_transformation_configs.items():
            self.logger.info(f"Loading transformation config: {transformation}")
            try:
                module_path, class_name = transformation.class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                transformer_class = getattr(module, class_name)
                params = vars(transformation)
                del params["class_path"]
                self.feature_transformations[name] = transformer_class(**params)
                self.logger.debug(f"Added transformation for feature: {name}")
            except Exception as e:
                self.logger.error(f"Failed to load transformer {transformation}: {e}")
                raise

    def output_dim(self) -> int:
        return sum([transformation.output_dim() for transformation in self.feature_transformations.values()])

    def forward(self, tabular_features: dict[str, torch.Tensor]) -> torch.Tensor:
        transformed_features = []
        total_dim = 0
        for name, feature in tabular_features.items():
            if name in self.feature_transformations:
                total_dim += feature.shape[1]
                transformed = self.feature_transformations[name](feature)
                transformed_features.append(transformed)
            else:
                self.logger.warn(f"No transformer found for feature: {name}")
        concatenated_features = torch.cat(transformed_features, dim=1)
        return concatenated_features


class TabularEmbeddingModel(nn.Module):
    """
    Model for encoding tabular data based on configurable feature transformations.
    """
    def __init__(self, config: MyConfig):
        """
        Args:
            config (MyConfig): Configuration object containing feature transformation definitions.
            device (str): Device to run the model on, e.g., "cpu" or "cuda".
        """
        super().__init__()
        self.logger = Logger(__name__)
        self.device = config.device
        self.model_config = config.model_config
        # Convert the feature transformations to a ordered dict by name
        feature_transformation_configs = {
            ft.name: ft for ft in self.model_config.feature_transformation_configs
        }
        # self.logger.info(f"Feature transformations: {feature_transformation_configs}")
        self.tabular_encoder = TabularEncoder(
            feature_transformation_configs=feature_transformation_configs
        ).to(self.device)

        mlp_input_dim = self.tabular_encoder.output_dim()
        self.logger.info(
            f"Num of feature transfomers {len(feature_transformation_configs)}. MLP input dimension: {mlp_input_dim}"
        )
        # Define the MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        ).to(self.device)

    def forward(self, tabular_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            tabular_features (dict[str, torch.Tensor]): Dictionary of tabular features as tensors.

        Returns:
            torch.Tensor: Encoded tabular features concatenated into a single tensor.
        """
        encoded_features = self.tabular_encoder(tabular_features).to(self.device)
        return self.mlp(encoded_features).to(self.device)


def init_model() -> TabularEmbeddingModel:
    """
    Initialize the TabularEmbeddingModel from a configuration file.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        TabularEmbeddingModel: Initialized model instance.
    """
    # Load the configuration
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(MyConfig, 'my_config')
    args = parser.parse_args()
    model_config = parser.instantiate_classes(args).my_config.model_config

    return TabularEmbeddingModel(config=model_config)


if __name__ == "__main__":
    model = init_model()
