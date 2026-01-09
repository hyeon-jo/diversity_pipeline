"""Main pipeline orchestration for video curation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .config import VideoEmbedderConfig, ClusteringConfig, AnalysisConfig
from .types import PipelineResults
from .embedder import VideoEmbedder
from .clusterer import GraphClusterer
from .analyzer import ClusterAnalyzer
from .captioning import CaptioningInterface, InternVLCaptioningInterface, MockCaptioningInterface

logger = logging.getLogger(__name__)


class VideoCurationPipeline:
    """
    Main orchestration class for the video curation pipeline.

    Combines all components:
    - VideoEmbedder for feature extraction (InternVL3.5-8B)
    - GraphClusterer for Leiden clustering
    - ClusterAnalyzer for analysis and void detection
    - CaptioningInterface for scenario descriptions (InternVL or Mock)
    """

    def __init__(
        self,
        embedder_config: Optional[VideoEmbedderConfig] = None,
        clustering_config: Optional[ClusteringConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
        captioning_interface: Optional[CaptioningInterface] = None,
        frame_base_dir: Optional[Union[str, Path]] = None,
        use_internvl_captioning: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            embedder_config: Configuration for video embedding.
            clustering_config: Configuration for graph clustering.
            analysis_config: Configuration for cluster analysis.
            captioning_interface: Interface for video captioning (overrides auto-creation).
            frame_base_dir: Base directory for frame directories (for InternVL captioning fallback).
            use_internvl_captioning: Whether to use InternVL for captioning (default: True).
        """
        self.embedder_config = embedder_config or VideoEmbedderConfig()
        self.embedder = VideoEmbedder(self.embedder_config)
        self.clusterer = GraphClusterer(clustering_config)
        self.analyzer = ClusterAnalyzer(analysis_config)

        # Captioning setup
        self._captioner = captioning_interface
        self._use_internvl_captioning = use_internvl_captioning
        self._frame_base_dir = Path(frame_base_dir) if frame_base_dir else None

    @property
    def captioner(self) -> CaptioningInterface:
        """
        Get the captioning interface (lazy loaded for InternVL).

        Returns:
            CaptioningInterface instance.
        """
        if self._captioner is not None:
            return self._captioner

        # If InternVL captioning is enabled and model is loaded, create InternVL captioner
        if self._use_internvl_captioning and self.embedder._is_loaded:
            self._captioner = InternVLCaptioningInterface(
                model=self.embedder.model,
                tokenizer=self.embedder.tokenizer,
                embedding_cache_dir=self.embedder_config.output_dir,
                frame_base_dir=self._frame_base_dir,
                input_size=self.embedder_config.input_size,
                num_frames=self.embedder_config.num_frames,
                torch_dtype=self.embedder_config.torch_dtype
            )
        else:
            # Fallback to mock captioner
            self._captioner = MockCaptioningInterface()

        return self._captioner

    def get_internvl_captioner(
        self,
        frame_base_dir: Optional[Union[str, Path]] = None
    ) -> InternVLCaptioningInterface:
        """
        Get or create an InternVL captioning interface.

        This method ensures the embedder model is loaded and creates
        an InternVL captioner that shares the model.

        Args:
            frame_base_dir: Override frame base directory.

        Returns:
            InternVLCaptioningInterface instance.
        """
        # Ensure model is loaded
        if not self.embedder._is_loaded:
            self.embedder.load_model()

        # Create InternVL captioner
        return InternVLCaptioningInterface(
            model=self.embedder.model,
            tokenizer=self.embedder.tokenizer,
            embedding_cache_dir=self.embedder_config.output_dir,
            frame_base_dir=frame_base_dir or self._frame_base_dir,
            input_size=self.embedder_config.input_size,
            num_frames=self.embedder_config.num_frames,
            torch_dtype=self.embedder_config.torch_dtype
        )

    def run(
        self,
        video_paths: list[str],
        generate_captions: bool = True
    ) -> PipelineResults:
        """
        Run the complete curation pipeline on a set of videos.

        Args:
            video_paths: List of paths to video files.
            generate_captions: Whether to generate cluster captions.

        Returns:
            Complete pipeline results.
        """
        logger.info(f"Starting pipeline with {len(video_paths)} videos")

        # Step 1: Extract embeddings
        logger.info("Step 1: Extracting video embeddings...")
        embeddings, failed_indices = self.embedder.extract_embeddings_batch(
            video_paths
        )

        # Filter out failed videos
        valid_paths = [
            p for i, p in enumerate(video_paths)
            if i not in failed_indices
        ]

        if len(failed_indices) > 0:
            logger.warning(f"{len(failed_indices)} videos failed to process")

        # Step 2: Graph clustering
        logger.info("Step 2: Performing Leiden clustering...")
        labels = self.clusterer.fit_predict(embeddings)

        # Step 3: Cluster analysis
        logger.info("Step 3: Analyzing clusters...")
        cluster_info, edge_cases, voids = self.analyzer.analyze(
            embeddings, labels, valid_paths
        )

        # Build representative videos mapping
        representative_videos = {
            cid: info.representative_path
            for cid, info in cluster_info.items()
            if info.representative_path is not None
        }

        # Step 4: Generate captions
        captions = {}
        if generate_captions and representative_videos:
            logger.info("Step 4: Generating cluster captions...")
            captions = self.captioner.generate_cluster_captions(
                representative_videos
            )

        logger.info("Pipeline complete!")

        return PipelineResults(
            embeddings=embeddings,
            cluster_labels=labels,
            cluster_info=cluster_info,
            representative_videos=representative_videos,
            edge_case_indices=edge_cases,
            void_cluster_ids=voids,
            captions=captions
        )

    def run_from_embeddings(
        self,
        embeddings: NDArray[np.float32],
        video_paths: Optional[list[str]] = None,
        generate_captions: bool = True
    ) -> PipelineResults:
        """
        Run pipeline from pre-computed embeddings (for demo/testing).

        Args:
            embeddings: Pre-computed embedding vectors.
            video_paths: Optional list of video paths.
            generate_captions: Whether to generate cluster captions.

        Returns:
            Complete pipeline results.
        """
        logger.info(f"Running pipeline with {len(embeddings)} embeddings")

        # Step 1: Graph clustering
        logger.info("Step 1: Performing Leiden clustering...")
        labels = self.clusterer.fit_predict(embeddings)

        # Step 2: Cluster analysis
        logger.info("Step 2: Analyzing clusters...")
        cluster_info, edge_cases, voids = self.analyzer.analyze(
            embeddings, labels, video_paths
        )

        # Build representative videos mapping
        representative_videos = {}
        for cid, info in cluster_info.items():
            if info.representative_path:
                representative_videos[cid] = info.representative_path
            else:
                # Use dummy path for demo
                representative_videos[cid] = f"video_{info.representative_idx}.mp4"

        # Step 3: Generate captions
        captions = {}
        if generate_captions:
            logger.info("Step 3: Generating cluster captions...")
            captions = self.captioner.generate_cluster_captions(
                representative_videos
            )

        logger.info("Pipeline complete!")

        return PipelineResults(
            embeddings=embeddings,
            cluster_labels=labels,
            cluster_info=cluster_info,
            representative_videos=representative_videos,
            edge_case_indices=edge_cases,
            void_cluster_ids=voids,
            captions=captions
        )
