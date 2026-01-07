"""
Core diversity metrics implementation.

References:
    [1] Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric
        for Machine Learning", ICLR 2023
    [2] Kynkäänniemi et al., "Improved Precision and Recall Metric for
        Assessing Generative Models", NeurIPS 2019
    [3] Naeem et al., "Reliable Fidelity and Diversity Metrics for
        Generative Models", ICML 2020
    [4] Xu et al., "On the Diversity of Autonomous Driving Datasets",
        CVPR 2022 Workshop
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

logger = logging.getLogger(__name__)


def vendi_score(
    embeddings: NDArray[np.float32],
    q: float = 1.0,
    kernel: str = "cosine",
    normalize: bool = True
) -> float:
    """
    Compute the Vendi Score - effective number of unique elements.

    The Vendi Score is defined as the exponential of the Shannon entropy
    of the eigenvalues of the similarity matrix, providing an intuitive
    measure of "effective diversity" - how many truly distinct items exist.

    VS = exp(-Σ λᵢ log(λᵢ))

    where λᵢ are the eigenvalues of the normalized similarity matrix.

    Reference:
        Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric
        for Machine Learning", ICLR 2023

    Args:
        embeddings: Array of shape (n_samples, embedding_dim).
        q: Order of diversity (q=1 gives Shannon entropy, q=2 gives Simpson).
        kernel: Similarity kernel ("cosine", "rbf", "linear").
        normalize: Whether to normalize the similarity matrix.

    Returns:
        Vendi Score (effective number of unique scenarios).
        Range: [1, n_samples]. Higher is more diverse.
    """
    n = len(embeddings)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    # Compute similarity matrix
    if kernel == "cosine":
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings_norm = embeddings / norms
        K = embeddings_norm @ embeddings_norm.T
        # Ensure values are in [0, 1] for cosine similarity
        K = (K + 1) / 2
    elif kernel == "rbf":
        # RBF kernel with automatic bandwidth
        K = rbf_kernel(embeddings)
    else:  # linear
        K = embeddings @ embeddings.T

    # Normalize the kernel matrix
    if normalize:
        K = K / n

    # Compute eigenvalues
    eigenvalues = linalg.eigvalsh(K)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    eigenvalues = eigenvalues / eigenvalues.sum()  # Normalize to sum to 1

    # Filter out near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 1.0

    # Compute Vendi Score based on order q
    if q == 1:
        # Shannon entropy: VS = exp(-Σ λᵢ log(λᵢ))
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        score = np.exp(entropy)
    elif q == 2:
        # Simpson diversity: VS = 1 / Σ λᵢ²
        score = 1.0 / np.sum(eigenvalues ** 2)
    else:
        # General Rényi entropy
        score = np.sum(eigenvalues ** q) ** (1 / (1 - q))

    return float(score)


def coverage_score(
    embeddings: NDArray[np.float32],
    reference_embeddings: NDArray[np.float32],
    k: int = 5,
    metric: str = "cosine"
) -> float:
    """
    Compute Coverage - proportion of reference distribution covered.

    Coverage measures what fraction of the reference dataset's manifold
    is "covered" by the evaluated dataset. A sample from the reference
    is considered covered if it has at least one neighbor from the
    evaluated dataset within a threshold distance.

    Reference:
        Kynkäänniemi et al., "Improved Precision and Recall Metric for
        Assessing Generative Models", NeurIPS 2019

    Args:
        embeddings: Evaluated dataset embeddings (n_samples, dim).
        reference_embeddings: Reference dataset embeddings (n_ref, dim).
        k: Number of neighbors for threshold computation.
        metric: Distance metric ("cosine" or "euclidean").

    Returns:
        Coverage score in [0, 1]. Higher means better coverage of reference.
    """
    if len(embeddings) == 0 or len(reference_embeddings) == 0:
        return 0.0

    # Compute k-NN distances within reference set to get threshold
    nn_ref = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn_ref.fit(reference_embeddings)
    distances_ref, _ = nn_ref.kneighbors(reference_embeddings)
    # Use k-th neighbor distance as threshold (skip self at index 0)
    thresholds = distances_ref[:, -1]

    # For each reference sample, check if any evaluated sample is within threshold
    nn_eval = NearestNeighbors(n_neighbors=1, metric=metric)
    nn_eval.fit(embeddings)
    distances_to_eval, _ = nn_eval.kneighbors(reference_embeddings)
    distances_to_eval = distances_to_eval[:, 0]

    # Count covered reference samples
    covered = np.sum(distances_to_eval <= thresholds)
    coverage = covered / len(reference_embeddings)

    return float(coverage)


def density_score(
    embeddings: NDArray[np.float32],
    reference_embeddings: Optional[NDArray[np.float32]] = None,
    k: int = 5,
    metric: str = "cosine"
) -> float:
    """
    Compute Density - average manifold density of the dataset.

    Density measures how many evaluated samples fall within the
    typical neighborhood of reference samples. Higher density indicates
    samples are concentrated in high-density regions of the reference.

    Reference:
        Naeem et al., "Reliable Fidelity and Diversity Metrics for
        Generative Models", ICML 2020

    Args:
        embeddings: Evaluated dataset embeddings.
        reference_embeddings: Reference dataset. If None, self-density is computed.
        k: Number of neighbors for density estimation.
        metric: Distance metric.

    Returns:
        Density score. Higher indicates samples in denser regions.
    """
    if len(embeddings) == 0:
        return 0.0

    if reference_embeddings is None:
        reference_embeddings = embeddings

    # Compute k-NN distances within reference set
    nn_ref = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn_ref.fit(reference_embeddings)
    distances_ref, _ = nn_ref.kneighbors(reference_embeddings)
    radii = distances_ref[:, -1]  # k-th neighbor distance as radius

    # For each evaluated sample, count how many reference spheres it falls into
    nn_eval = NearestNeighbors(n_neighbors=min(k, len(reference_embeddings)), metric=metric)
    nn_eval.fit(reference_embeddings)
    distances_to_ref, indices = nn_eval.kneighbors(embeddings)

    # Count spheres each sample falls into
    counts = np.zeros(len(embeddings))
    for i, (dists, idxs) in enumerate(zip(distances_to_ref, indices)):
        for dist, idx in zip(dists, idxs):
            if dist <= radii[idx]:
                counts[i] += 1

    density = np.mean(counts) / k

    return float(density)


def feature_space_coverage(
    embeddings: NDArray[np.float32],
    n_bins: int = 10,
    use_pca: bool = True,
    n_components: int = 10,
    reference_embeddings: Optional[NDArray[np.float32]] = None
) -> dict[str, float]:
    """
    Compute Feature Space Coverage - grid-based coverage in embedding space.

    Divides the embedding space into a grid and measures what proportion
    of cells are occupied. Provides both absolute and relative coverage.

    Reference:
        Xu et al., "On the Diversity of Autonomous Driving Datasets",
        CVPR 2022 Workshop

    Args:
        embeddings: Evaluated dataset embeddings.
        n_bins: Number of bins per dimension.
        use_pca: Whether to reduce dimensionality with PCA first.
        n_components: Number of PCA components if use_pca=True.
        reference_embeddings: Optional reference for relative coverage.

    Returns:
        Dictionary with coverage metrics:
        - absolute_coverage: Proportion of grid cells occupied (0-100%)
        - relative_coverage: Coverage relative to reference (0-100%), if provided
        - occupied_cells: Number of occupied cells
        - total_cells: Total number of cells
    """
    from sklearn.decomposition import PCA

    if len(embeddings) == 0:
        return {
            "absolute_coverage": 0.0,
            "relative_coverage": 0.0,
            "occupied_cells": 0,
            "total_cells": 0
        }

    # Reduce dimensionality if needed
    if use_pca:
        n_comp = min(n_components, embeddings.shape[1], len(embeddings) - 1)
        pca = PCA(n_components=n_comp)

        if reference_embeddings is not None:
            # Fit on combined data for fair comparison
            combined = np.vstack([embeddings, reference_embeddings])
            pca.fit(combined)
            embeddings_reduced = pca.transform(embeddings)
            reference_reduced = pca.transform(reference_embeddings)
        else:
            embeddings_reduced = pca.fit_transform(embeddings)
            reference_reduced = None
    else:
        embeddings_reduced = embeddings
        reference_reduced = reference_embeddings

    n_dims = embeddings_reduced.shape[1]

    # Compute grid bounds
    if reference_reduced is not None:
        combined = np.vstack([embeddings_reduced, reference_reduced])
    else:
        combined = embeddings_reduced

    mins = combined.min(axis=0)
    maxs = combined.max(axis=0)

    # Discretize embeddings to grid cells
    def to_grid_cell(emb):
        normalized = (emb - mins) / (maxs - mins + 1e-8)
        cell = (normalized * n_bins).astype(int)
        cell = np.clip(cell, 0, n_bins - 1)
        return tuple(cell)

    # Count occupied cells
    occupied = set()
    for emb in embeddings_reduced:
        occupied.add(to_grid_cell(emb))

    # Calculate theoretical max cells (considering actual data spread)
    # In high dimensions, theoretical max is too large, so we use reference
    if reference_reduced is not None:
        reference_occupied = set()
        for emb in reference_reduced:
            reference_occupied.add(to_grid_cell(emb))
        total_cells = len(reference_occupied)
        relative_coverage = len(occupied & reference_occupied) / max(len(reference_occupied), 1)
    else:
        # Use number of samples as upper bound for occupied cells
        total_cells = min(len(embeddings), n_bins ** min(n_dims, 3))
        relative_coverage = 0.0

    absolute_coverage = len(occupied) / max(total_cells, 1)

    return {
        "absolute_coverage": float(absolute_coverage) * 100,  # As percentage
        "relative_coverage": float(relative_coverage) * 100 if reference_reduced is not None else None,
        "occupied_cells": len(occupied),
        "total_cells": total_cells
    }


def scenario_entropy(
    cluster_labels: NDArray[np.int32],
    normalize: bool = True
) -> dict[str, float]:
    """
    Compute Scenario Distribution Entropy.

    Measures how uniformly distributed samples are across scenario types
    (clusters). Maximum entropy occurs when all scenarios have equal
    representation.

    Args:
        cluster_labels: Cluster assignment for each sample.
        normalize: Whether to normalize by maximum possible entropy.

    Returns:
        Dictionary with entropy metrics:
        - entropy: Shannon entropy of cluster distribution
        - normalized_entropy: Entropy / log(n_clusters) (0-100%)
        - balance_score: 1 - normalized Gini coefficient (0-100%)
        - n_clusters: Number of unique clusters
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    n_clusters = len(unique)

    if n_clusters <= 1:
        return {
            "entropy": 0.0,
            "normalized_entropy": 100.0 if n_clusters == 1 else 0.0,
            "balance_score": 100.0 if n_clusters == 1 else 0.0,
            "n_clusters": n_clusters
        }

    # Compute probabilities
    probs = counts / counts.sum()

    # Shannon entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(n_clusters)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    # Gini coefficient for balance
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])
    balance_score = 1 - gini

    return {
        "entropy": float(entropy),
        "normalized_entropy": float(normalized_entropy) * 100,
        "balance_score": float(balance_score) * 100,
        "n_clusters": n_clusters
    }


def rarity_score(
    embeddings: NDArray[np.float32],
    reference_embeddings: Optional[NDArray[np.float32]] = None,
    k: int = 10,
    percentile: float = 5.0,
    metric: str = "cosine"
) -> dict[str, Union[float, list[int]]]:
    """
    Compute Rarity Score - detection of rare/unique scenarios.

    Identifies samples that are far from their neighbors, indicating
    unique or underrepresented scenarios. Useful for finding edge cases
    and gaps in dataset coverage.

    Args:
        embeddings: Evaluated dataset embeddings.
        reference_embeddings: Reference for relative rarity. If None, self-rarity.
        k: Number of neighbors for distance computation.
        percentile: Percentile threshold for "rare" classification.
        metric: Distance metric.

    Returns:
        Dictionary with rarity metrics:
        - mean_rarity: Average isolation score (higher = more rare scenarios)
        - rare_sample_ratio: Percentage of samples classified as rare
        - rare_indices: Indices of rare samples
        - rarity_scores: Per-sample rarity scores
    """
    if len(embeddings) == 0:
        return {
            "mean_rarity": 0.0,
            "rare_sample_ratio": 0.0,
            "rare_indices": [],
            "rarity_scores": []
        }

    target = reference_embeddings if reference_embeddings is not None else embeddings
    k_actual = min(k, len(target) - 1)

    if k_actual < 1:
        return {
            "mean_rarity": 0.0,
            "rare_sample_ratio": 0.0,
            "rare_indices": [],
            "rarity_scores": []
        }

    # Compute k-NN distances
    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric=metric)
    nn.fit(target)
    distances, _ = nn.kneighbors(embeddings)

    # Skip self-connection if using self-reference
    if reference_embeddings is None:
        distances = distances[:, 1:]  # Skip self

    # Rarity score = mean distance to k nearest neighbors
    rarity_scores = distances.mean(axis=1)

    # Normalize to [0, 1] range
    if rarity_scores.max() > rarity_scores.min():
        rarity_scores_norm = (rarity_scores - rarity_scores.min()) / (rarity_scores.max() - rarity_scores.min())
    else:
        rarity_scores_norm = np.zeros_like(rarity_scores)

    # Identify rare samples
    threshold = np.percentile(rarity_scores, 100 - percentile)
    rare_indices = np.where(rarity_scores >= threshold)[0].tolist()

    return {
        "mean_rarity": float(rarity_scores_norm.mean()) * 100,
        "rare_sample_ratio": float(len(rare_indices) / len(embeddings)) * 100,
        "rare_indices": rare_indices,
        "rarity_scores": rarity_scores_norm.tolist()
    }


def intra_cluster_diversity(
    embeddings: NDArray[np.float32],
    cluster_labels: NDArray[np.int32],
    metric: str = "cosine"
) -> dict[str, Union[float, dict[int, float]]]:
    """
    Compute Intra-cluster Diversity.

    Measures diversity within each cluster. Low intra-cluster diversity
    indicates homogeneous clusters (good clustering), while high values
    indicate diverse samples within clusters.

    Args:
        embeddings: Sample embeddings.
        cluster_labels: Cluster assignments.
        metric: Distance metric.

    Returns:
        Dictionary with:
        - mean_diversity: Average within-cluster diversity (0-100%)
        - per_cluster: Dictionary of per-cluster diversity scores
        - cluster_variances: Per-cluster embedding variance
    """
    unique_labels = np.unique(cluster_labels)
    per_cluster_diversity = {}
    cluster_variances = {}

    for label in unique_labels:
        mask = cluster_labels == label
        cluster_embeddings = embeddings[mask]

        if len(cluster_embeddings) < 2:
            per_cluster_diversity[int(label)] = 0.0
            cluster_variances[int(label)] = 0.0
            continue

        # Compute pairwise distances within cluster
        if metric == "cosine":
            norms = np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normalized = cluster_embeddings / norms
            similarities = normalized @ normalized.T
            distances = 1 - similarities
        else:
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(cluster_embeddings, metric=metric))

        # Mean pairwise distance (excluding diagonal)
        n = len(cluster_embeddings)
        mean_dist = distances.sum() / (n * (n - 1)) if n > 1 else 0

        per_cluster_diversity[int(label)] = float(mean_dist)

        # Variance of embeddings
        variance = np.var(cluster_embeddings, axis=0).mean()
        cluster_variances[int(label)] = float(variance)

    # Weight by cluster size
    sizes = np.array([np.sum(cluster_labels == l) for l in unique_labels])
    diversities = np.array([per_cluster_diversity[int(l)] for l in unique_labels])
    weighted_mean = np.average(diversities, weights=sizes)

    return {
        "mean_diversity": float(weighted_mean) * 100,
        "per_cluster": per_cluster_diversity,
        "cluster_variances": cluster_variances
    }


def compute_all_metrics(
    embeddings: NDArray[np.float32],
    cluster_labels: NDArray[np.int32],
    reference_embeddings: Optional[NDArray[np.float32]] = None,
    k_neighbors: int = 5
) -> dict[str, Union[float, dict]]:
    """
    Compute all diversity metrics at once.

    Args:
        embeddings: Sample embeddings.
        cluster_labels: Cluster assignments.
        reference_embeddings: Optional reference dataset for comparison.
        k_neighbors: k parameter for k-NN based metrics.

    Returns:
        Dictionary containing all computed metrics.
    """
    logger.info("Computing all diversity metrics...")

    results = {}

    # Vendi Score
    logger.info("  Computing Vendi Score...")
    results["vendi_score"] = vendi_score(embeddings)

    # Scenario Entropy
    logger.info("  Computing Scenario Entropy...")
    results["scenario_entropy"] = scenario_entropy(cluster_labels)

    # Intra-cluster Diversity
    logger.info("  Computing Intra-cluster Diversity...")
    results["intra_cluster_diversity"] = intra_cluster_diversity(
        embeddings, cluster_labels
    )

    # Rarity Score
    logger.info("  Computing Rarity Score...")
    results["rarity"] = rarity_score(
        embeddings,
        reference_embeddings=reference_embeddings,
        k=k_neighbors
    )

    # Feature Space Coverage
    logger.info("  Computing Feature Space Coverage...")
    results["feature_space_coverage"] = feature_space_coverage(
        embeddings,
        reference_embeddings=reference_embeddings
    )

    # Reference-dependent metrics
    if reference_embeddings is not None:
        logger.info("  Computing Coverage Score...")
        results["coverage_score"] = coverage_score(
            embeddings, reference_embeddings, k=k_neighbors
        ) * 100  # As percentage

        logger.info("  Computing Density Score...")
        results["density_score"] = density_score(
            embeddings, reference_embeddings, k=k_neighbors
        ) * 100

    logger.info("All metrics computed.")

    return results
