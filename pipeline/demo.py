"""Demo utilities for the video curation pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import PipelineResults


def generate_synthetic_embeddings(
    n_samples: int = 500,
    embedding_dim: int = 1280,
    n_clusters: int = 10,
    noise_level: float = 0.3,
    seed: int = 42
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Generate synthetic embeddings for demonstration.

    Creates embeddings with known cluster structure including:
    - Dense clusters (well-represented scenarios)
    - Sparse clusters (voids)
    - Micro-clusters (edge cases)
    - Outliers

    Args:
        n_samples: Total number of samples to generate.
        embedding_dim: Dimension of embedding vectors.
        n_clusters: Number of clusters to create.
        noise_level: Amount of noise to add.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (embeddings, ground_truth_labels).
    """
    rng = np.random.default_rng(seed)

    embeddings = []
    labels = []

    # Generate cluster centers on unit hypersphere
    centers = rng.standard_normal((n_clusters, embedding_dim))
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Assign different sizes to clusters
    # Most clusters are medium, some sparse, some micro
    cluster_sizes = []
    remaining = n_samples

    for i in range(n_clusters):
        if i < 2:  # 2 micro-clusters
            size = rng.integers(1, 3)
        elif i < 4:  # 2 sparse clusters
            size = rng.integers(10, 20)
        else:  # Regular clusters
            size = max(1, remaining // (n_clusters - i))

        size = min(size, remaining)
        cluster_sizes.append(size)
        remaining -= size

    # Distribute any remaining samples
    if remaining > 0:
        cluster_sizes[-1] += remaining

    # Generate samples for each cluster
    for cluster_id, (center, size) in enumerate(zip(centers, cluster_sizes)):
        # Add noise around center
        noise = rng.standard_normal((size, embedding_dim)) * noise_level
        cluster_samples = center + noise

        # Normalize to unit sphere
        cluster_samples = cluster_samples / np.linalg.norm(
            cluster_samples, axis=1, keepdims=True
        )

        embeddings.append(cluster_samples)
        labels.extend([cluster_id] * size)

    # Add some outliers
    n_outliers = max(5, n_samples // 50)
    outliers = rng.standard_normal((n_outliers, embedding_dim))
    outliers = outliers / np.linalg.norm(outliers, axis=1, keepdims=True)
    embeddings.append(outliers)
    labels.extend([n_clusters] * n_outliers)  # New cluster for outliers

    embeddings = np.vstack(embeddings).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)

    return embeddings, labels


def print_results_summary(results: PipelineResults) -> None:
    """Print a human-readable summary of pipeline results."""
    print("\n" + "=" * 60)
    print("VIDEO CURATION PIPELINE RESULTS")
    print("=" * 60)

    print(f"\nTotal videos processed: {len(results.embeddings)}")
    print(f"Number of clusters: {len(results.cluster_info)}")
    print(f"Edge cases identified: {len(results.edge_case_indices)}")
    print(f"Void clusters (under-represented): {len(results.void_cluster_ids)}")

    print("\n" + "-" * 60)
    print("CLUSTER DETAILS")
    print("-" * 60)

    for cid, info in sorted(results.cluster_info.items()):
        status = []
        if info.is_micro_cluster:
            status.append("MICRO-CLUSTER")
        if info.is_low_density:
            status.append("LOW-DENSITY")
        status_str = f" [{', '.join(status)}]" if status else ""

        print(f"\nCluster {cid}{status_str}:")
        print(f"  Size: {info.size} videos")
        print(f"  Density: {info.density:.4f}")
        print(f"  Representative: {info.representative_path or f'idx={info.representative_idx}'}")

        if cid in results.captions:
            print(f"  Caption: {results.captions[cid]}")

    if results.edge_case_indices:
        print("\n" + "-" * 60)
        print("EDGE CASES (Outliers & Micro-cluster members)")
        print("-" * 60)
        print(f"Indices: {results.edge_case_indices[:20]}")
        if len(results.edge_case_indices) > 20:
            print(f"  ... and {len(results.edge_case_indices) - 20} more")

    if results.void_cluster_ids:
        print("\n" + "-" * 60)
        print("VOID CLUSTERS (Under-represented scenarios)")
        print("-" * 60)
        print(f"Cluster IDs: {results.void_cluster_ids}")
        print("These clusters may indicate missing training data!")

    print("\n" + "=" * 60)
