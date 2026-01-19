"""
BEV Diversity Pipeline

A clean implementation for extracting visual embeddings from frame images
using InternVL3.5-8B and computing the Vendi Score Family diversity metrics.
"""

from bev_diversity.config import PipelineConfig, load_config
from bev_diversity.embedder.extractor import ImageEmbedder
from bev_diversity.metrics.vendi_family import VendiScoreFamily

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "load_config",
    "ImageEmbedder",
    "VendiScoreFamily",
]
