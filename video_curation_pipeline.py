#!/usr/bin/env python3
"""
Video-Centric Data Curation Pipeline

A comprehensive pipeline for analyzing diversity in driving video clips using:
- InternVL3.5-8B for feature extraction and captioning
- Leiden Graph Clustering for unsupervised scenario discovery
- Edge case and void detection for identifying missing data
- Cached embedding support for efficient captioning

This file provides backward compatibility by re-exporting all components
from the modularized pipeline package.

Author: AI Engineer
License: MIT
"""

# Re-export all public API from the modularized pipeline package
from pipeline import (
    # Configuration classes
    VideoEmbedderConfig,
    ClusteringConfig,
    AnalysisConfig,
    # Type definitions
    ClusterInfo,
    PipelineResults,
    # Core components
    VideoEmbedder,
    GraphClusterer,
    ClusterAnalyzer,
    # Captioning interfaces
    CaptioningInterface,
    InternVLCaptioningInterface,
    MockCaptioningInterface,
    # Main pipeline
    VideoCurationPipeline,
    # Demo utilities
    generate_synthetic_embeddings,
    print_results_summary,
)

# Import CLI for main entry point
from pipeline.cli import main

# Configure logging
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    main()
