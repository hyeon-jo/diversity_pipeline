from __future__ import annotations
import logging
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray
from .config import ClusteringConfig

logger = logging.getLogger(__name__)

class GraphClusterer:
    """
    Construct k-NN graph and perform Leiden community detection.

    This approach captures manifold structure better than K-Means for
    complex driving scenarios with non-convex cluster shapes.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the GraphClusterer.

        Args:
            config: Configuration for clustering. Uses defaults if None.
        """
        self.config = config or ClusteringConfig()
        self.knn_graph_ = None
        self.adjacency_matrix_ = None

    def _build_knn_graph(
        self,
        embeddings: NDArray[np.float32]
    ) -> Any:
        """
        Build approximate k-NN graph from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).

        Returns:
            Sparse adjacency matrix.
        """
        n_samples = embeddings.shape[0]
        k = min(self.config.k_neighbors, n_samples - 1)

        logger.info(f"Building k-NN graph with k={k}")

        # Try pynndescent for approximate NN (faster for large datasets)
        try:
            from pynndescent import NNDescent

            index = NNDescent(
                embeddings,
                metric=self.config.metric,
                n_neighbors=k + 1,  # +1 because it includes self
                random_state=self.config.random_state
            )
            indices, distances = index.neighbor_graph

            # Remove self-connections
            indices = indices[:, 1:]
            distances = distances[:, 1:]

        except ImportError:
            logger.info("pynndescent not available, using sklearn")
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(
                n_neighbors=k,
                metric=self.config.metric,
                algorithm="auto"
            )
            nn.fit(embeddings)
            distances, indices = nn.kneighbors(embeddings)

        # Build sparse adjacency matrix
        from scipy import sparse

        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = indices.flatten()

        # Use similarity (1 - distance for cosine)
        if self.config.metric == "cosine":
            similarities = 1 - distances.flatten()
        else:
            # For euclidean, use Gaussian kernel
            sigma = np.median(distances)
            similarities = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))

        # Create symmetric adjacency matrix
        adjacency = sparse.csr_matrix(
            (similarities, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        # Make symmetric
        adjacency = adjacency + adjacency.T
        adjacency.data = np.minimum(adjacency.data, 1.0)

        self.adjacency_matrix_ = adjacency
        return adjacency

    def _leiden_clustering(
        self,
        adjacency: Any
    ) -> NDArray[np.int32]:
        """
        Apply Leiden algorithm for community detection.

        Args:
            adjacency: Sparse adjacency matrix.

        Returns:
            Array of cluster labels.
        """
        logger.info("Running Leiden clustering...")

        # Try leidenalg (most reliable)
        try:
            import igraph as ig
            import leidenalg

            # Convert to igraph
            sources, targets = adjacency.nonzero()
            weights = adjacency[sources, targets].A1

            g = ig.Graph(
                n=adjacency.shape[0],
                edges=list(zip(sources.tolist(), targets.tolist())),
                directed=False
            )
            g.es["weight"] = weights.tolist()

            # Run Leiden
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=self.config.resolution,
                n_iterations=self.config.n_iterations,
                seed=self.config.random_state
            )

            return np.array(partition.membership, dtype=np.int32)

        except ImportError:
            logger.warning("leidenalg not available, trying cdlib")

        # Fallback to cdlib
        try:
            import networkx as nx
            from cdlib import algorithms

            G = nx.from_scipy_sparse_array(adjacency)

            # cdlib Leiden
            communities = algorithms.leiden(
                G,
                weights="weight",
                resolution_parameter=self.config.resolution
            )

            # Convert to labels
            labels = np.zeros(adjacency.shape[0], dtype=np.int32)
            for cluster_id, members in enumerate(communities.communities):
                for member in members:
                    labels[member] = cluster_id

            return labels

        except ImportError:
            logger.warning("cdlib not available, falling back to Louvain")

        # Final fallback to sklearn spectral clustering
        from sklearn.cluster import SpectralClustering

        n_clusters = max(2, int(np.sqrt(adjacency.shape[0] / 2)))

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=self.config.random_state
        )

        return clustering.fit_predict(adjacency.toarray()).astype(np.int32)

    def fit_predict(
        self,
        embeddings: NDArray[np.float32]
    ) -> NDArray[np.int32]:
        """
        Build graph and perform clustering.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).

        Returns:
            Array of cluster labels.
        """
        adjacency = self._build_knn_graph(embeddings)
        labels = self._leiden_clustering(adjacency)

        n_clusters = len(np.unique(labels))
        logger.info(f"Found {n_clusters} clusters")

        return labels