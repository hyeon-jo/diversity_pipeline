"""
k-NN Network graph visualization.

Shows the similarity structure of video embeddings as a network:
- Nodes represent videos
- Edges represent k-NN connections
- Node colors indicate cluster membership
- Layout algorithms reveal cluster structure
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from .config import VisualizerConfig

if TYPE_CHECKING:
    from video_curation_pipeline import PipelineResults

logger = logging.getLogger(__name__)


class NetworkGraphPlotter:
    """
    Creates network graph visualizations of the k-NN similarity structure.

    Features:
    - Interactive Pyvis HTML graphs
    - Static NetworkX/Matplotlib plots
    - Multiple layout algorithms
    - Cluster-based node coloring
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize the network graph plotter.

        Args:
            config: Visualization configuration.
        """
        self.config = config or VisualizerConfig()
        self._graph = None
        self._layout = None

    def build_knn_graph(
        self,
        embeddings: NDArray[np.float32],
        k: int = 15
    ) -> "networkx.Graph":
        """
        Build a k-NN graph from embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            k: Number of nearest neighbors.

        Returns:
            NetworkX Graph object.
        """
        import networkx as nx
        from sklearn.neighbors import NearestNeighbors

        logger.info(f"Building k-NN graph with k={k}...")

        # Compute k-NN
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        # Build graph
        self._graph = nx.Graph()

        n_samples = len(embeddings)
        self._graph.add_nodes_from(range(n_samples))

        # Add edges (skip self-connections at index 0)
        for i in range(n_samples):
            for j, dist in zip(indices[i, 1:], distances[i, 1:]):
                if i < j:  # Avoid duplicate edges
                    self._graph.add_edge(i, j, weight=1 - dist)  # Similarity weight

        logger.info(f"Graph has {self._graph.number_of_nodes()} nodes, "
                    f"{self._graph.number_of_edges()} edges")

        return self._graph

    def compute_layout(
        self,
        layout_type: Optional[str] = None
    ) -> dict[int, tuple[float, float]]:
        """
        Compute node positions using a layout algorithm.

        Args:
            layout_type: Layout algorithm ("spring", "kamada_kawai", "circular").

        Returns:
            Dictionary mapping node IDs to (x, y) positions.
        """
        import networkx as nx

        if self._graph is None:
            raise ValueError("Graph not built. Call build_knn_graph first.")

        layout_type = layout_type or self.config.network_layout

        logger.info(f"Computing {layout_type} layout...")

        if layout_type == "spring":
            self._layout = nx.spring_layout(
                self._graph,
                k=1 / np.sqrt(self._graph.number_of_nodes()),
                iterations=50,
                seed=42
            )
        elif layout_type == "kamada_kawai":
            self._layout = nx.kamada_kawai_layout(self._graph)
        elif layout_type == "circular":
            self._layout = nx.circular_layout(self._graph)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

        return self._layout

    def plot_interactive(
        self,
        results: "PipelineResults",
        video_names: Optional[list[str]] = None,
        k: int = 15,
        title: str = "k-NN Similarity Network",
        save_path: Optional[str] = None,
        height: str = "800px",
        width: str = "100%"
    ) -> str:
        """
        Create an interactive Pyvis network visualization.

        Args:
            results: Pipeline results.
            video_names: Optional list of video names for node labels.
            k: Number of nearest neighbors for graph.
            title: Network title.
            save_path: Path to save HTML file.
            height: Canvas height.
            width: Canvas width.

        Returns:
            Path to saved HTML file.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis required for interactive network graphs. "
                "Install with: pip install pyvis"
            )

        # Build graph if not already done
        if self._graph is None:
            self.build_knn_graph(results.embeddings, k)

        # Create Pyvis network
        net = Network(
            height=height,
            width=width,
            bgcolor="#ffffff",
            font_color="#333333",
            heading=title
        )

        # Configure physics
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": true,
                    "iterations": 100
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100
            }
        }
        """)

        # Prepare colors
        import plotly.express as px
        n_clusters = len(set(results.cluster_labels))
        color_palette = px.colors.qualitative.Set3 if n_clusters <= 12 else px.colors.qualitative.Light24

        # Prepare node info
        labels = results.cluster_labels
        if video_names is None:
            video_names = [f"Video_{i}" for i in range(len(labels))]

        # Add nodes
        for i, (name, label) in enumerate(zip(video_names, labels)):
            cluster_info = results.cluster_info.get(label)

            # Determine node properties
            color = color_palette[label % len(color_palette)]
            size = 10

            # Highlight special nodes
            if i in results.edge_case_indices:
                border_color = self.config.edge_case_color
                border_width = 3
            else:
                border_color = "#333333"
                border_width = 1

            # Mark representatives
            is_rep = any(
                info.representative_idx == i
                for info in results.cluster_info.values()
            )
            if is_rep:
                size = 20
                shape = "star"
            else:
                shape = "dot"

            # Build hover title
            title_text = f"<b>{name}</b><br>"
            title_text += f"Cluster: {label}<br>"
            if cluster_info:
                title_text += f"Cluster Size: {cluster_info.size}<br>"
            if i in results.edge_case_indices:
                title_text += "<span style='color:red'>Edge Case</span><br>"
            if label in results.void_cluster_ids:
                title_text += "<span style='color:teal'>Void Cluster</span><br>"

            net.add_node(
                i,
                label=f"C{label}",
                title=title_text,
                color=color,
                size=size,
                shape=shape,
                borderWidth=border_width,
                borderWidthSelected=border_width + 2
            )

        # Add edges
        for u, v, data in self._graph.edges(data=True):
            weight = data.get("weight", 0.5)
            net.add_edge(
                u, v,
                width=weight * 2,
                color=f"rgba(150, 150, 150, {self.config.edge_alpha})"
            )

        # Save
        if save_path is None:
            save_path = Path(self.config.output_dir) / "network_graph.html"

        net.save_graph(str(save_path))
        logger.info(f"Saved interactive network to {save_path}")

        return str(save_path)

    def plot_static(
        self,
        results: "PipelineResults",
        video_names: Optional[list[str]] = None,
        k: int = 15,
        title: str = "k-NN Similarity Network",
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 12)
    ) -> "matplotlib.figure.Figure":
        """
        Create a static Matplotlib network visualization.

        Args:
            results: Pipeline results.
            video_names: Optional list of video names.
            k: Number of nearest neighbors for graph.
            title: Network title.
            save_path: Path to save image.
            figsize: Figure size.

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        # Build graph if not already done
        if self._graph is None:
            self.build_knn_graph(results.embeddings, k)

        # Compute layout if not already done
        if self._layout is None:
            self.compute_layout()

        labels = results.cluster_labels
        n_clusters = len(set(labels))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get colormap
        cmap = plt.get_cmap(self.config.colormap)
        node_colors = [cmap(labels[i] / max(n_clusters - 1, 1)) for i in range(len(labels))]

        # Determine node sizes
        node_sizes = []
        for i in range(len(labels)):
            is_rep = any(
                info.representative_idx == i
                for info in results.cluster_info.values()
            )
            if is_rep:
                node_sizes.append(self.config.node_size_scale * 4)
            elif i in results.edge_case_indices:
                node_sizes.append(self.config.node_size_scale * 2)
            else:
                node_sizes.append(self.config.node_size_scale)

        # Draw edges
        nx.draw_networkx_edges(
            self._graph,
            self._layout,
            alpha=self.config.edge_alpha,
            edge_color="#cccccc",
            ax=ax
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self._graph,
            self._layout,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax
        )

        # Highlight edge cases with red borders
        if results.edge_case_indices:
            edge_case_pos = {i: self._layout[i] for i in results.edge_case_indices}
            nx.draw_networkx_nodes(
                self._graph.subgraph(results.edge_case_indices),
                edge_case_pos,
                node_color="none",
                node_size=[node_sizes[i] * 1.5 for i in results.edge_case_indices],
                edgecolors=self.config.edge_case_color,
                linewidths=2,
                ax=ax
            )

        # Draw representative labels
        rep_indices = [info.representative_idx for info in results.cluster_info.values()]
        rep_labels = {i: f"C{labels[i]}" for i in rep_indices}
        rep_pos = {i: self._layout[i] for i in rep_indices}

        nx.draw_networkx_labels(
            self._graph.subgraph(rep_indices),
            rep_pos,
            rep_labels,
            font_size=8,
            font_weight="bold",
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=10, label=f"{n_clusters} Clusters"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=self.config.edge_case_color, markersize=10,
                   markeredgewidth=2, label=f"{len(results.edge_case_indices)} Edge Cases"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=15, label="Representative"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

        plt.tight_layout()

        # Save if requested
        if save_path is None and self.config.save_png:
            save_path = Path(self.config.output_dir) / "network_graph.png"

        if save_path:
            fig.savefig(str(save_path), dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved static network to {save_path}")

        return fig

    def get_graph_metrics(self) -> dict:
        """
        Compute graph metrics.

        Returns:
            Dictionary of graph metrics.
        """
        import networkx as nx

        if self._graph is None:
            raise ValueError("Graph not built. Call build_knn_graph first.")

        metrics = {
            "n_nodes": self._graph.number_of_nodes(),
            "n_edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph),
            "avg_clustering": nx.average_clustering(self._graph),
        }

        # Compute connected components
        components = list(nx.connected_components(self._graph))
        metrics["n_components"] = len(components)
        metrics["largest_component_size"] = max(len(c) for c in components)

        # Degree statistics
        degrees = [d for _, d in self._graph.degree()]
        metrics["avg_degree"] = np.mean(degrees)
        metrics["max_degree"] = max(degrees)
        metrics["min_degree"] = min(degrees)

        return metrics
