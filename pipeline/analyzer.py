"""Cluster analysis and edge case discovery."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import AnalysisConfig
from .types import ClusterInfo

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """
    Analyze clusters to find representatives, edge cases, and voids.

    - Computes cluster centroids and finds representative samples
    - Identifies micro-clusters (edge cases)
    - Detects sparse regions (voids/under-represented scenarios)
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the ClusterAnalyzer.

        Args:
            config: Configuration for analysis. Uses defaults if None.
        """
        self.config = config or AnalysisConfig()

    def _compute_cluster_centroids(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32]
    ) -> dict[int, NDArray[np.float32]]:
        """
        Compute centroid for each cluster.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.

        Returns:
            Dictionary mapping cluster_id to centroid vector.
        """
        centroids = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            centroid = cluster_embeddings.mean(axis=0)
            # Normalize centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[int(label)] = centroid

        return centroids

    def _compute_cluster_density(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: dict[int, NDArray[np.float32]]
    ) -> dict[int, float]:
        """
        Compute density for each cluster (inverse of avg distance to centroid).

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            centroids: Dictionary of cluster centroids.

        Returns:
            Dictionary mapping cluster_id to density score.
        """
        densities = {}

        for cluster_id, centroid in centroids.items():
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]

            # Compute distances to centroid
            distances = np.linalg.norm(
                cluster_embeddings - centroid, axis=1
            )

            avg_distance = distances.mean() if len(distances) > 0 else 1.0
            # Density is inverse of average distance
            densities[cluster_id] = 1.0 / (avg_distance + 1e-8)

        return densities

    def _find_representatives(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: dict[int, NDArray[np.float32]],
        video_paths: Optional[list[str]] = None
    ) -> dict[int, tuple[int, Optional[str]]]:
        """
        Find the most representative sample for each cluster.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            centroids: Dictionary of cluster centroids.
            video_paths: Optional list of video paths.

        Returns:
            Dictionary mapping cluster_id to (sample_idx, video_path).
        """
        representatives = {}

        for cluster_id, centroid in centroids.items():
            mask = labels == cluster_id
            indices = np.where(mask)[0]
            cluster_embeddings = embeddings[mask]

            # Find closest to centroid
            distances = np.linalg.norm(
                cluster_embeddings - centroid, axis=1
            )
            closest_idx = indices[np.argmin(distances)]

            path = video_paths[closest_idx] if video_paths else None
            representatives[cluster_id] = (int(closest_idx), path)

        return representatives

    def _identify_edge_cases(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        centroids: dict[int, NDArray[np.float32]],
        cluster_sizes: dict[int, int]
    ) -> list[int]:
        """
        Identify edge case samples (outliers and micro-cluster members).

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            centroids: Dictionary of cluster centroids.
            cluster_sizes: Dictionary of cluster sizes.

        Returns:
            List of edge case sample indices.
        """
        edge_cases = set()

        # 1. Members of micro-clusters
        micro_clusters = {
            cid for cid, size in cluster_sizes.items()
            if size < self.config.micro_cluster_threshold
        }

        for idx, label in enumerate(labels):
            if label in micro_clusters:
                edge_cases.add(idx)

        # 2. Outliers: samples far from their cluster centroid
        distances_to_centroid = []
        for idx, (embedding, label) in enumerate(zip(embeddings, labels)):
            centroid = centroids[int(label)]
            dist = np.linalg.norm(embedding - centroid)
            distances_to_centroid.append((idx, dist))

        # Find outliers based on distance percentile
        all_distances = [d for _, d in distances_to_centroid]
        threshold = np.percentile(
            all_distances,
            self.config.outlier_distance_percentile
        )

        for idx, dist in distances_to_centroid:
            if dist > threshold:
                edge_cases.add(idx)

        return sorted(list(edge_cases))

    def _identify_voids(
        self,
        densities: dict[int, float],
        cluster_sizes: dict[int, int]
    ) -> list[int]:
        """
        Identify void clusters (sparse, under-represented scenarios).

        Args:
            densities: Dictionary of cluster densities.
            cluster_sizes: Dictionary of cluster sizes.

        Returns:
            List of void cluster IDs.
        """
        voids = []

        density_values = list(densities.values())
        threshold = np.percentile(
            density_values,
            self.config.low_density_percentile
        )

        for cluster_id, density in densities.items():
            # Low density AND not a micro-cluster
            # (micro-clusters are handled separately as edge cases)
            if (density < threshold and
                cluster_sizes[cluster_id] >= self.config.micro_cluster_threshold):
                voids.append(cluster_id)

        return voids

    def analyze(
        self,
        embeddings: NDArray[np.float32],
        labels: NDArray[np.int32],
        video_paths: Optional[list[str]] = None
    ) -> tuple[dict[int, ClusterInfo], list[int], list[int]]:
        """
        Perform complete cluster analysis.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Cluster labels for each sample.
            video_paths: Optional list of video paths.

        Returns:
            Tuple of (cluster_info dict, edge_case_indices, void_cluster_ids).
        """
        logger.info("Analyzing clusters...")

        # Compute cluster statistics
        unique_labels = np.unique(labels)
        cluster_sizes = {
            int(label): int(np.sum(labels == label))
            for label in unique_labels
        }

        centroids = self._compute_cluster_centroids(embeddings, labels)
        densities = self._compute_cluster_density(embeddings, labels, centroids)
        representatives = self._find_representatives(
            embeddings, labels, centroids, video_paths
        )

        # Identify edge cases and voids
        edge_cases = self._identify_edge_cases(
            embeddings, labels, centroids, cluster_sizes
        )
        voids = self._identify_voids(densities, cluster_sizes)

        # Build cluster info
        micro_clusters = {
            cid for cid, size in cluster_sizes.items()
            if size < self.config.micro_cluster_threshold
        }

        cluster_info = {}
        for cluster_id in unique_labels:
            cid = int(cluster_id)
            rep_idx, rep_path = representatives[cid]

            cluster_info[cid] = ClusterInfo(
                cluster_id=cid,
                size=cluster_sizes[cid],
                centroid=centroids[cid],
                density=densities[cid],
                representative_idx=rep_idx,
                representative_path=rep_path,
                is_micro_cluster=cid in micro_clusters,
                is_low_density=cid in voids
            )

        logger.info(f"Found {len(edge_cases)} edge cases")
        logger.info(f"Found {len(voids)} void clusters")

        return cluster_info, edge_cases, voids
