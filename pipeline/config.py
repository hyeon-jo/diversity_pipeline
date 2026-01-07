"""Configuration dataclasses for the video curation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


def _get_device() -> str:
    """Get the best available device."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class VideoEmbedderConfig:
    """Configuration for InternVL3.5-8B embedding extraction."""
    model_name: str = "OpenGVLab/InternVL3_5-8B"  # InternVL3.5-8B (8.5B params)
    num_frames: int = 16  # Number of frames to sample (num_segments)
    input_size: int = 448  # InternVL input resolution
    max_num: int = 1  # Max tiles per frame (1 for video, up to 12 for images)
    device: str = field(default_factory=_get_device)
    normalize_embeddings: bool = True
    frame_sample_strategy: str = "uniform"  # "uniform" or "random"
    trust_remote_code: bool = True  # Required for OpenGVLab models
    output_dir: str = "output"  # Directory to save embeddings
    use_flash_attn: bool = True  # Use flash attention for efficiency
    torch_dtype: str = "bfloat16"  # "bfloat16", "float16", or "float32"


@dataclass
class ClusteringConfig:
    """Configuration for graph construction and Leiden clustering."""
    k_neighbors: int = 15
    metric: str = "cosine"
    resolution: float = 1.0  # Leiden resolution parameter
    n_iterations: int = -1  # -1 for unlimited iterations
    random_state: int = 42


@dataclass
class AnalysisConfig:
    """Configuration for cluster analysis and void detection."""
    micro_cluster_threshold: int = 3  # Clusters smaller than this are edge cases
    low_density_percentile: float = 10.0  # Bottom X% density = void
    outlier_distance_percentile: float = 95.0  # Top X% distance = outlier
