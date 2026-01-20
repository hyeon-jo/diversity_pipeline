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
class VideoConfig:
    """Configuration for video embedding extraction."""

    num_frames: int = 16  # Number of frames to sample per video
    frame_sample_strategy: str = "uniform"  # "uniform" or "random"
    pooling_strategy: str = "mean"  # "mean" or "max"
    normalize_embeddings: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if self.frame_sample_strategy not in ["uniform", "random"]:
            raise ValueError(f"Invalid frame_sample_strategy: {self.frame_sample_strategy}")
        if self.pooling_strategy not in ["mean", "max"]:
            raise ValueError(f"Invalid pooling_strategy: {self.pooling_strategy}")


@dataclass
class InputConfig:
    """Configuration for input data structure."""

    mode: str = "video_folders"  # "video_folders" or "single_folder"
    frames_per_video: Optional[int] = None  # For single_folder mode grouping

    def __post_init__(self):
        """Validate configuration."""
        if self.mode not in ["video_folders", "single_folder"]:
            raise ValueError(f"Invalid input mode: {self.mode}")
        if self.mode == "single_folder" and self.frames_per_video is not None:
            if self.frames_per_video < 1:
                raise ValueError("frames_per_video must be >= 1")


@dataclass
class OutputConfig:
    """Configuration for output and memory optimization."""

    save_individual_embeddings: bool = True  # Save each video embedding as separate file
    embedding_subdir: str = "embeddings"  # Subdirectory for individual embeddings
    skip_existing: bool = True  # Skip videos that already have embeddings (resume support)
    save_combined_embeddings: bool = True  # Also save all embeddings as single file


@dataclass
class PipelineConfig:
    """Top-level configuration combining all settings."""

    model: Optional[ModelConfig]
    embedding: Optional[EmbeddingConfig]
    vendi: VendiConfig
    video: Optional[VideoConfig] = None  # Video mode config
    input_config: Optional[InputConfig] = None  # Input structure config
    output_config: Optional[OutputConfig] = None  # Output and memory config
    input_dir: Optional[str] = None  # Root directory containing images/videos
    embedding_path: Optional[str] = None  # Path to pre-computed embeddings
    embedding_dir: Optional[str] = None  # Directory containing individual embedding files
    output_dir: str = "./output"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png"]
    )

    def __post_init__(self):
        """Validate configuration."""
        # Must specify one of: input_dir, embedding_path, or embedding_dir
        sources = [self.input_dir, self.embedding_path, self.embedding_dir]
        specified_sources = [s for s in sources if s is not None]

        if len(specified_sources) == 0:
            raise ValueError("Must specify one of: input_dir, embedding_path, or embedding_dir")

        if len(specified_sources) > 1:
            raise ValueError("Cannot specify multiple sources. Choose one of: input_dir, embedding_path, embedding_dir")

        # If input_dir is specified, validate it and require model config
        if self.input_dir:
            if not Path(self.input_dir).exists():
                raise ValueError(f"Input directory does not exist: {self.input_dir}")
            if not self.image_extensions:
                raise ValueError("image_extensions cannot be empty")
            if self.model is None:
                raise ValueError("model config is required when using input_dir")

            # For video mode, video config is required; for image mode, embedding config is required
            if self.video is None and self.embedding is None:
                raise ValueError("Either video or embedding config is required when using input_dir")

        # If embedding_path is specified, validate it
        if self.embedding_path:
            if not Path(self.embedding_path).exists():
                raise ValueError(f"Embedding file does not exist: {self.embedding_path}")
            if not self.embedding_path.endswith(('.npy', '.npz')):
                raise ValueError("embedding_path must be a .npy or .npz file")

        # If embedding_dir is specified, validate it
        if self.embedding_dir:
            if not Path(self.embedding_dir).exists():
                raise ValueError(f"Embedding directory does not exist: {self.embedding_dir}")

    def is_embedding_only_mode(self) -> bool:
        """Check if running in embedding-only mode (no extraction)."""
        return self.embedding_path is not None or self.embedding_dir is not None

    def is_video_mode(self) -> bool:
        """Check if running in video mode (frame pooling)."""
        return self.video is not None


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

    # Extract top-level paths
    embedding_path = data.get("embedding_path")
    embedding_dir = data.get("embedding_dir")
    input_dir = data.get("input_dir")

    # Parse model config (optional if embedding_path/embedding_dir is provided)
    model_config = None
    if "model" in data and data["model"]:
        model_data = data.get("model", {})
        model_config = ModelConfig(**model_data)

    # Parse embedding config (for image mode)
    embedding_config = None
    if "embedding" in data and data["embedding"]:
        embedding_data = data.get("embedding", {})
        embedding_config = EmbeddingConfig(**embedding_data)

    # Parse video config (for video mode)
    video_config = None
    if "video" in data and data["video"]:
        video_data = data.get("video", {})
        video_config = VideoConfig(**video_data)

    # Parse input config (for video mode folder structure)
    input_config = None
    if "input" in data and data["input"]:
        input_data = data.get("input", {})
        input_config = InputConfig(**input_data)

    # Parse output config (for memory optimization)
    output_config = None
    if "output" in data and data["output"]:
        output_data = data.get("output", {})
        output_config = OutputConfig(**output_data)

    # Parse vendi config (always required)
    vendi_data = data.get("vendi", {})
    vendi_config = VendiConfig(**vendi_data)

    # Parse top-level config
    pipeline_config = PipelineConfig(
        model=model_config,
        embedding=embedding_config,
        vendi=vendi_config,
        video=video_config,
        input_config=input_config,
        output_config=output_config,
        input_dir=input_dir,
        embedding_path=embedding_path,
        embedding_dir=embedding_dir,
        output_dir=data.get("output_dir", "./output"),
        image_extensions=data.get("image_extensions", [".jpg", ".jpeg", ".png"]),
    )

    return pipeline_config
