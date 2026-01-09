"""
Video Curation Pipeline

A comprehensive pipeline for video analysis using:
- InternVL3.5-8B for embedding extraction
- Leiden clustering for scenario discovery
- Edge case and void detection
"""

from .config import VideoEmbedderConfig, ClusteringConfig, AnalysisConfig
from .types import ClusterInfo, PipelineResults
from .embedder import VideoEmbedder
from .clusterer import GraphClusterer
from .analyzer import ClusterAnalyzer
from .captioning import CaptioningInterface, InternVLCaptioningInterface, MockCaptioningInterface
from .pipeline import VideoCurationPipeline
from .demo import generate_synthetic_embeddings, print_results_summary

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "VideoEmbedderConfig",
    "ClusteringConfig",
    "AnalysisConfig",
    # Types
    "ClusterInfo",
    "PipelineResults",
    # Core components
    "VideoEmbedder",
    "GraphClusterer",
    "ClusterAnalyzer",
    # Captioning
    "CaptioningInterface",
    "InternVLCaptioningInterface",
    "MockCaptioningInterface",
    # Pipeline
    "VideoCurationPipeline",
    # Demo utilities
    "generate_synthetic_embeddings",
    "print_results_summary",
]
