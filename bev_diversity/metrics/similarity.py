"""Similarity matrix computation for embeddings."""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import rbf_kernel


class SimilarityMatrix:
    """
    Compute and manage similarity matrices from embeddings.

    Supports multiple kernel types:
    - cosine: Cosine similarity (normalized dot product)
    - rbf: Radial Basis Function / Gaussian kernel
    - linear: Linear kernel (dot product)
    """

    def __init__(self, kernel: str = "cosine", rbf_gamma: float = 1.0):
        """
        Initialize similarity matrix computer.

        Args:
            kernel: Kernel type ("cosine", "rbf", "linear")
            rbf_gamma: Gamma parameter for RBF kernel (only used if kernel="rbf")
        """
        if kernel not in ["cosine", "rbf", "linear"]:
            raise ValueError(f"Invalid kernel: {kernel}")

        self.kernel = kernel
        self.rbf_gamma = rbf_gamma

    def compute(self, embeddings: NDArray) -> NDArray:
        """
        Compute similarity matrix K from embeddings.

        Args:
            embeddings: Embedding matrix [N, D]

        Returns:
            Similarity matrix [N, N]
        """
        n = len(embeddings)

        if self.kernel == "cosine":
            # Cosine similarity
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            embeddings_norm = embeddings / norms

            # Compute cosine similarity
            K = embeddings_norm @ embeddings_norm.T

            # Map from [-1, 1] to [0, 1] for valid similarity matrix
            K = (K + 1) / 2

        elif self.kernel == "rbf":
            # RBF / Gaussian kernel: exp(-gamma * ||x - x'||^2)
            K = rbf_kernel(embeddings, gamma=self.rbf_gamma)

        elif self.kernel == "linear":
            # Linear kernel: x @ x'
            K = embeddings @ embeddings.T

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        # Ensure symmetry (numerical stability)
        K = (K + K.T) / 2

        # Ensure non-negative
        K = np.maximum(K, 0)

        return K

    def normalize(self, K: NDArray) -> NDArray:
        """
        Normalize similarity matrix by n (as per Vendi Score paper: K/n).

        Args:
            K: Similarity matrix [N, N]

        Returns:
            Normalized similarity matrix [N, N]
        """
        n = len(K)
        return K / n

    def compute_normalized(self, embeddings: NDArray) -> NDArray:
        """
        Compute and normalize similarity matrix in one step.

        Args:
            embeddings: Embedding matrix [N, D]

        Returns:
            Normalized similarity matrix [N, N]
        """
        K = self.compute(embeddings)
        return self.normalize(K)
