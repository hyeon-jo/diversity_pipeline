"""
Generalized Vendi Score Family implementation.

Based on the paper "Cousins Of The Vendi Score: A Family Of Similarity-Based
Diversity Metrics For Science And Machine Learning" (AISTATS 2024).
"""

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigvalsh

from bev_diversity.metrics.similarity import SimilarityMatrix


class VendiScoreFamily:
    """
    Compute generalized Vendi Scores with different order q values.

    The Vendi Score Family extends the original Vendi Score (q=1) to
    provide different levels of sensitivity to rare or common items.

    Mathematical formulas:
    - For q = 1 (Shannon entropy):
        VS_1 = exp(-Σ λ_i log(λ_i))

    - For q ≠ 1, ∞:
        VS_q = exp((1/(1-q)) * log(Σ λ_i^q))

    - For q = ∞:
        VS_∞ = 1 / max(λ_i)

    Where λ_i are the eigenvalues of the normalized similarity matrix K/n.

    Properties:
    - Smaller q values (e.g., 0.1, 0.5) are more sensitive to rare items
    - Larger q values (e.g., 2.0, ∞) are more sensitive to common items
    - VS_∞ ≤ VS_2 ≤ VS_1 ≤ VS_0.5 ≤ VS_0.1 (monotonicity)
    - Higher scores indicate greater diversity
    - Scores represent "effective number of unique scenarios"
    """

    def __init__(self, kernel: str = "cosine", rbf_gamma: float = 1.0):
        """
        Initialize Vendi Score Family computer.

        Args:
            kernel: Kernel type for similarity ("cosine", "rbf", "linear")
            rbf_gamma: Gamma parameter for RBF kernel
        """
        self.similarity = SimilarityMatrix(kernel=kernel, rbf_gamma=rbf_gamma)

    def compute_eigenvalues(self, K_normalized: NDArray) -> NDArray:
        """
        Compute eigenvalues of normalized similarity matrix.

        Args:
            K_normalized: Normalized similarity matrix K/n [N, N]

        Returns:
            Eigenvalues sorted in descending order
        """
        # Use eigvalsh for symmetric matrices (more efficient and stable)
        eigenvalues = eigvalsh(K_normalized)

        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Filter out near-zero and negative eigenvalues (numerical stability)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        return eigenvalues

    def vendi_score_q1(self, eigenvalues: NDArray) -> float:
        """
        Original Vendi Score (q=1, Shannon entropy).

        Formula: VS_1 = exp(-Σ λ_i log(λ_i))

        Args:
            eigenvalues: Eigenvalues of K/n

        Returns:
            Vendi Score (effective number of unique items)
        """
        # Filter positive eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # Shannon entropy: -Σ λ_i log(λ_i)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues + 1e-12))

        # Vendi Score is exponential of entropy
        return float(np.exp(entropy))

    def vendi_score_q(self, eigenvalues: NDArray, q: float) -> float:
        """
        Generalized Vendi Score for order q ≠ 1, ∞.

        Formula: VS_q = exp((1/(1-q)) * log(Σ λ_i^q))

        Args:
            eigenvalues: Eigenvalues of K/n
            q: Order parameter (controls sensitivity)

        Returns:
            Vendi Score of order q
        """
        # Special case: q = 1
        if np.isclose(q, 1.0):
            return self.vendi_score_q1(eigenvalues)

        # Special case: q = infinity
        if np.isinf(q):
            return self.vendi_score_infinity(eigenvalues)

        # Filter positive eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        # General formula: exp((1/(1-q)) * log(Σ λ_i^q))
        power_sum = np.sum(eigenvalues ** q)

        if power_sum <= 0:
            return 1.0  # Minimum diversity

        exponent = (1.0 / (1.0 - q)) * np.log(power_sum)

        return float(np.exp(exponent))

    def vendi_score_infinity(self, eigenvalues: NDArray) -> float:
        """
        Vendi Score of infinite order.

        Formula: VS_∞ = 1 / max(λ_i)

        This score is most sensitive to duplicates and common items.

        Args:
            eigenvalues: Eigenvalues of K/n

        Returns:
            Vendi Score of infinite order
        """
        # Filter positive eigenvalues
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) == 0:
            return 1.0

        max_eigenvalue = np.max(eigenvalues)

        return float(1.0 / max_eigenvalue)

    def compute_all(
        self,
        embeddings: NDArray,
        q_values: List[float],
        include_infinity: bool = True,
    ) -> Dict[str, float]:
        """
        Compute Vendi Scores for all specified q values.

        Args:
            embeddings: Embedding matrix [N, D]
            q_values: List of q values to compute
            include_infinity: Whether to also compute VS_∞

        Returns:
            Dictionary mapping "VS_q={value}" to score

        Example:
            >>> embeddings = np.random.randn(100, 512)
            >>> vendi = VendiScoreFamily(kernel="cosine")
            >>> scores = vendi.compute_all(embeddings, [0.1, 0.5, 1.0, 2.0])
            >>> print(scores)
            {'VS_q=0.1': 89.23, 'VS_q=0.5': 67.45, 'VS_q=1.0': 45.67, 'VS_q=2.0': 28.91, 'VS_q=inf': 12.34}
        """
        # Compute similarity matrix
        K = self.similarity.compute(embeddings)
        K_norm = self.similarity.normalize(K)

        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues(K_norm)

        # Compute scores for all q values
        results = {}
        for q in q_values:
            key = f"VS_q={q}"
            results[key] = self.vendi_score_q(eigenvalues, q)

        # Optionally add infinity score
        if include_infinity:
            results["VS_q=inf"] = self.vendi_score_infinity(eigenvalues)

        return results

    def compute_single(self, embeddings: NDArray, q: float) -> float:
        """
        Compute Vendi Score for a single q value.

        Args:
            embeddings: Embedding matrix [N, D]
            q: Order parameter

        Returns:
            Vendi Score of order q
        """
        # Compute similarity matrix
        K = self.similarity.compute(embeddings)
        K_norm = self.similarity.normalize(K)

        # Compute eigenvalues
        eigenvalues = self.compute_eigenvalues(K_norm)

        # Compute score
        return self.vendi_score_q(eigenvalues, q)
