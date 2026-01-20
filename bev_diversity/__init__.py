"""
BEV Diversity Pipeline

A clean implementation for extracting visual embeddings from images or videos
using InternVL3.5-8B and computing the Vendi Score Family diversity metrics.

Supports three modes:
1. IMAGE MODE: Extract embeddings from individual images
2. VIDEO MODE: Extract video embeddings using frame pooling
3. EMBEDDING-ONLY MODE: Compute Vendi scores from pre-computed embeddings
"""

from bev_diversity.config import (
    PipelineConfig,
    ModelConfig,
    EmbeddingConfig,
    VideoConfig,
    InputConfig,
    OutputConfig,
    VendiConfig,
    load_config,
)
from bev_diversity.embedder.extractor import ImageEmbedder
from bev_diversity.embedder.video_embedder import VideoEmbedder, load_embeddings_from_dir
from bev_diversity.metrics.vendi_family import VendiScoreFamily

__version__ = "2.0.0"
__all__ = [
    # Config classes
    "PipelineConfig",
    "ModelConfig",
    "EmbeddingConfig",
    "VideoConfig",
    "InputConfig",
    "OutputConfig",
    "VendiConfig",
    "load_config",
    # Embedders
    "ImageEmbedder",
    "VideoEmbedder",
    "load_embeddings_from_dir",
    # Metrics
    "VendiScoreFamily",
]
