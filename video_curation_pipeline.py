#!/usr/bin/env python3
"""
Video-Centric Data Curation Pipeline

A comprehensive pipeline for analyzing diversity in driving video clips using:
- InternVL3.5-8B for feature extraction and captioning
- Leiden Graph Clustering for unsupervised scenario discovery
- Edge case and void detection for identifying missing data
- Cached embedding support for efficient captioning

Author: AI Engineer
License: MIT
"""

from pipeline import (
    VideoEmbedderConfig,
    ClusteringConfig,
    AnalysisConfig,
    ClusterInfo,
    PipelineResults,
    VideoEmbedder,
    GraphClusterer,
    ClusterAnalyzer,
    CaptioningInterface,
    InternVLCaptioningInterface,
    MockCaptioningInterface,
    VideoCurationPipeline,
    generate_synthetic_embeddings,
    print_results_summary
)
from pipeline.cli import main

if __name__ == "__main__":
    main()