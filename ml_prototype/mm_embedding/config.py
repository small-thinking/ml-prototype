from dataclasses import dataclass
from typing import Optional


@dataclass
class Column:
    name: str
    type: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass
class DataModuleConfig:
    folder_path: str
    batch_size: int = 32
    val_ratio: float = 0.1
    columns: Optional[list[Column]] = None


@dataclass
class FeatureTransformationConfig:
    """
    Represents a single feature transformation configuration.
    """
    class_path: str
    name: str


@dataclass
class IdentityConfig(FeatureTransformationConfig):
    """
    Represents the configuration for the identity transformation.
    """
    name: str = "identity"


@dataclass
class FeatureScalingConfig(FeatureTransformationConfig):
    """Conduct min-max normalization on the input features."""
    name: str = "min_max"
    min_val: float = 0.0
    max_val: float = 1.0


@dataclass
class OneHotEncodingConfig(FeatureTransformationConfig):
    """
    Represents the configuration for the one-hot encoding transformation.
    """
    name: str = "one_hot"
    num_classes: Optional[int] = None


@dataclass
class EmbeddingTransformer(FeatureTransformationConfig):
    """
    Represents the configuration for the embedding transformer.
    """
    name: str
    num_embeddings: int
    embedding_dim: int = 16
    num_layers: int = 1
    num_heads: int = 4
    ff_dim: int = 32
    dropout: float = 0.1


@dataclass
class AugmentConfig:
    """
    Represents the configuration for data augmentation.
    """
    augment_numerical: bool = False
    noise_std: float = 0.1  # Standard deviation of Gaussian noise
    augment_categorical: bool = False
    change_prob: float = 0.1  # Probability of changing a categorical feature
    change_delta: int = 1  # Maximum change in a categorical feature


@dataclass
class TrainingConfig:
    """
    Represents the training configuration.
    """
    epochs: int
    learning_rate: float
    augment_config: AugmentConfig
    use_wandb: bool = False


@dataclass
class ModelConfig:
    """
    Represents the model configuration, including feature transformations.
    """
    transformer_layers: int
    embedding_dim: int
    ff_dim: int
    feature_transformation_configs: list[
        IdentityConfig | OneHotEncodingConfig | EmbeddingTransformer | FeatureScalingConfig
    ]


@dataclass
class MyConfig:
    device: str
    data_module: DataModuleConfig
    training_config: TrainingConfig
    model_config: ModelConfig
    use_wandb: bool = False
