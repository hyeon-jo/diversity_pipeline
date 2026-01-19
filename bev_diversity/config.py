"""Configuration classes and YAML loading for BEV diversity pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for InternVL model loading."""

    model_path: str  # Required: LOCAL path to model directory
    input_size: int = 448
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", "float32"
    device: str = "cuda"  # "cuda" or "cpu"
    use_flash_attn: bool = True
    trust_remote_code: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not self.model_path:
            raise ValueError("model_path is required")
        if not Path(self.model_path).exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")
        if self.torch_dtype not in ["bfloat16", "float16", "float32"]:
            raise ValueError(f"Invalid torch_dtype: {self.torch_dtype}")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""

    batch_size: int = 4
    normalize_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


@dataclass
class VendiConfig:
    """Configuration for Vendi Score computation."""

    q_values: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0])
    include_infinity: bool = True
    kernel: str = "cosine"  # "cosine", "rbf", "linear"
    rbf_gamma: float = 1.0  # Only used if kernel="rbf"

    def __post_init__(self):
        """Validate configuration."""
        if not self.q_values:
            raise ValueError("q_values cannot be empty")
        if self.kernel not in ["cosine", "rbf", "linear"]:
            raise ValueError(f"Invalid kernel: {self.kernel}")
        if self.rbf_gamma <= 0:
            raise ValueError("rbf_gamma must be > 0")


@dataclass
class PipelineConfig:
    """Top-level configuration combining all settings."""

    model: Optional[ModelConfig]
    embedding: Optional[EmbeddingConfig]
    vendi: VendiConfig
    input_dir: Optional[str] = None  # Root directory containing images (for extraction)
    embedding_path: Optional[str] = None  # Path to pre-computed embeddings (.npy file)
    output_dir: str = "./output"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )

    def __post_init__(self):
        """Validate configuration."""
        # Either input_dir or embedding_path must be specified
        if not self.input_dir and not self.embedding_path:
            raise ValueError("Either input_dir or embedding_path must be specified")

        # Cannot specify both
        if self.input_dir and self.embedding_path:
            raise ValueError("Cannot specify both input_dir and embedding_path. Choose one.")

        # If input_dir is specified, validate it and require model config
        if self.input_dir:
            if not Path(self.input_dir).exists():
                raise ValueError(f"Input directory does not exist: {self.input_dir}")
            if not self.image_extensions:
                raise ValueError("image_extensions cannot be empty")
            if self.model is None:
                raise ValueError("model config is required when using input_dir")
            if self.embedding is None:
                raise ValueError("embedding config is required when using input_dir")

        # If embedding_path is specified, validate it
        if self.embedding_path:
            if not Path(self.embedding_path).exists():
                raise ValueError(f"Embedding file does not exist: {self.embedding_path}")
            if not self.embedding_path.endswith(('.npy', '.npz')):
                raise ValueError("embedding_path must be a .npy or .npz file")

    def is_embedding_only_mode(self) -> bool:
        """Check if running in embedding-only mode (no extraction)."""
        return self.embedding_path is not None


def load_config(yaml_path: str) -> PipelineConfig:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        PipelineConfig instance

    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If configuration is invalid

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config.model.model_path)
    """
    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid YAML format: expected dictionary")

    # Check if embedding_path is provided (embedding-only mode)
    embedding_path = data.get("embedding_path")
    input_dir = data.get("input_dir")

    # Parse model config (optional if embedding_path is provided)
    model_config = None
    if "model" in data and data["model"]:
        model_data = data.get("model", {})
        model_config = ModelConfig(**model_data)

    # Parse embedding config (optional if embedding_path is provided)
    embedding_config = None
    if "embedding" in data and data["embedding"]:
        embedding_data = data.get("embedding", {})
        embedding_config = EmbeddingConfig(**embedding_data)

    # Parse vendi config (always required)
    vendi_data = data.get("vendi", {})
    vendi_config = VendiConfig(**vendi_data)

    # Parse top-level config
    pipeline_config = PipelineConfig(
        model=model_config,
        embedding=embedding_config,
        vendi=vendi_config,
        input_dir=input_dir,
        embedding_path=embedding_path,
        output_dir=data.get("output_dir", "./output"),
        image_extensions=data.get("image_extensions", [".jpg", ".jpeg", ".png"]),
    )

    return pipeline_config
