"""
Embedding visualization using t-SNE/UMAP dimensionality reduction.

Provides interactive 2D scatter plots of video embeddings with:
- Cluster-based coloring
- Edge case highlighting
- Void cluster markers
- Representative video indicators
- Hover tooltips with video info
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


class EmbeddingPlotter:
    """
    Creates 2D embedding visualizations using UMAP or t-SNE.

    Features:
    - Interactive Plotly charts with hover info
    - Static Matplotlib plots for reports
    - Cluster coloring with special markers for edge cases/voids
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize the embedding plotter.

        Args:
            config: Visualization configuration.
        """
        self.config = config or VisualizerConfig()
        self._reducer = None
        self._embeddings_2d = None

    def reduce_dimensions(
        self,
        embeddings: NDArray[np.float32],
        method: Optional[str] = None
    ) -> NDArray[np.float32]:
        """
        Reduce high-dimensional embeddings to 2D.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            method: Reduction method ("umap" or "tsne"). Uses config default if None.

        Returns:
            Array of shape (n_samples, 2).
        """
        method = method or self.config.reduction_method

        logger.info(f"Reducing dimensions using {method.upper()}...")

        if method == "umap":
            return self._reduce_umap(embeddings)
        elif method == "tsne":
            return self._reduce_tsne(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

    def _reduce_umap(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reduce dimensions using UMAP."""
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn required for UMAP reduction. "
                "Install with: pip install umap-learn"
            )

        self._reducer = umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            n_components=2,
            metric="cosine",
            random_state=42
        )

        self._embeddings_2d = self._reducer.fit_transform(embeddings)
        return self._embeddings_2d

    def _reduce_tsne(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """Reduce dimensions using t-SNE."""
        from sklearn.manifold import TSNE

        self._reducer = TSNE(
            n_components=2,
            perplexity=min(self.config.perplexity, len(embeddings) - 1),
            metric="cosine",
            random_state=42,
            init="pca"
        )

        self._embeddings_2d = self._reducer.fit_transform(embeddings)
        return self._embeddings_2d

    def plot_interactive(
        self,
        results: "PipelineResults",
        video_names: Optional[list[str]] = None,
        title: str = "Video Embedding Space",
        save_path: Optional[str] = None
    ) -> "plotly.graph_objects.Figure":
        """
        Create an interactive Plotly scatter plot.

        Args:
            results: Pipeline results containing embeddings and cluster info.
            video_names: Optional list of video names for hover text.
            title: Plot title.
            save_path: Path to save HTML file. Uses config default if None.

        Returns:
            Plotly Figure object.
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            raise ImportError(
                "plotly required for interactive plots. "
                "Install with: pip install plotly"
            )

        # Reduce dimensions if not already done
        if self._embeddings_2d is None:
            self._embeddings_2d = self.reduce_dimensions(results.embeddings)

        embeddings_2d = self._embeddings_2d
        labels = results.cluster_labels
        n_clusters = len(set(labels))

        # Create color mapping
        colors = px.colors.qualitative.Set3 if n_clusters <= 12 else px.colors.qualitative.Light24
        color_map = {i: colors[i % len(colors)] for i in range(n_clusters)}

        # Prepare hover text
        if video_names is None:
            video_names = [f"Video_{i}" for i in range(len(labels))]

        hover_texts = []
        for i, (name, label) in enumerate(zip(video_names, labels)):
            cluster_info = results.cluster_info.get(label)
            size = cluster_info.size if cluster_info else "N/A"
            is_edge = "Yes" if i in results.edge_case_indices else "No"
            is_void = "Yes" if label in results.void_cluster_ids else "No"

            hover_texts.append(
                f"<b>{name}</b><br>"
                f"Cluster: {label}<br>"
                f"Cluster Size: {size}<br>"
                f"Edge Case: {is_edge}<br>"
                f"Void Cluster: {is_void}"
            )

        # Create figure
        fig = go.Figure()

        # Plot regular points by cluster
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            is_void = cluster_id in results.void_cluster_ids

            # Determine marker style
            marker_symbol = "diamond" if is_void else "circle"
            marker_color = self.config.void_cluster_color if is_void else color_map[cluster_id]

            cluster_info = results.cluster_info.get(cluster_id)
            cluster_name = f"Cluster {cluster_id}"
            if is_void:
                cluster_name += " (Void)"

            fig.add_trace(go.Scatter(
                x=embeddings_2d[mask, 0],
                y=embeddings_2d[mask, 1],
                mode="markers",
                name=cluster_name,
                marker=dict(
                    size=self.config.point_size,
                    color=marker_color,
                    symbol=marker_symbol,
                    line=dict(width=0.5, color="white")
                ),
                text=[hover_texts[i] for i in np.where(mask)[0]],
                hoverinfo="text"
            ))

        # Highlight edge cases
        if results.edge_case_indices:
            edge_mask = np.zeros(len(labels), dtype=bool)
            edge_mask[results.edge_case_indices] = True

            fig.add_trace(go.Scatter(
                x=embeddings_2d[edge_mask, 0],
                y=embeddings_2d[edge_mask, 1],
                mode="markers",
                name="Edge Cases",
                marker=dict(
                    size=self.config.point_size + 4,
                    color="rgba(0,0,0,0)",
                    line=dict(width=2, color=self.config.edge_case_color)
                ),
                text=[hover_texts[i] for i in results.edge_case_indices],
                hoverinfo="text"
            ))

        # Mark representative videos
        rep_indices = [
            info.representative_idx
            for info in results.cluster_info.values()
        ]

        if rep_indices:
            fig.add_trace(go.Scatter(
                x=embeddings_2d[rep_indices, 0],
                y=embeddings_2d[rep_indices, 1],
                mode="markers",
                name="Representatives",
                marker=dict(
                    size=self.config.point_size + 6,
                    color=self.config.representative_color,
                    symbol="star",
                    line=dict(width=1, color="black")
                ),
                text=[hover_texts[i] for i in rep_indices],
                hoverinfo="text"
            ))

        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title=f"{self.config.reduction_method.upper()} Dimension 1",
            yaxis_title=f"{self.config.reduction_method.upper()} Dimension 2",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02
            ),
            hovermode="closest",
            template="plotly_white",
            width=1200,
            height=800
        )

        # Save if requested
        if save_path is None and self.config.save_html:
            save_path = Path(self.config.output_dir) / "embedding_plot.html"

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Saved interactive plot to {save_path}")

        return fig

    def plot_static(
        self,
        results: "PipelineResults",
        video_names: Optional[list[str]] = None,
        title: str = "Video Embedding Space",
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 10)
    ) -> "matplotlib.figure.Figure":
        """
        Create a static Matplotlib scatter plot.

        Args:
            results: Pipeline results containing embeddings and cluster info.
            video_names: Optional list of video names.
            title: Plot title.
            save_path: Path to save PNG file.
            figsize: Figure size in inches.

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Reduce dimensions if not already done
        if self._embeddings_2d is None:
            self._embeddings_2d = self.reduce_dimensions(results.embeddings)

        embeddings_2d = self._embeddings_2d
        labels = results.cluster_labels
        n_clusters = len(set(labels))

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get colormap
        cmap = plt.get_cmap(self.config.colormap)
        colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

        # Plot each cluster
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            is_void = cluster_id in results.void_cluster_ids

            marker = "D" if is_void else "o"
            color = self.config.void_cluster_color if is_void else colors[cluster_id % len(colors)]

            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                marker=marker,
                s=self.config.point_size ** 2,
                alpha=0.7,
                label=f"Cluster {cluster_id}" if not is_void else f"Cluster {cluster_id} (Void)"
            )

        # Highlight edge cases with red circles
        if results.edge_case_indices:
            ax.scatter(
                embeddings_2d[results.edge_case_indices, 0],
                embeddings_2d[results.edge_case_indices, 1],
                facecolors="none",
                edgecolors=self.config.edge_case_color,
                s=(self.config.point_size + 4) ** 2,
                linewidths=2,
                label="Edge Cases"
            )

        # Mark representatives with stars
        rep_indices = [info.representative_idx for info in results.cluster_info.values()]
        if rep_indices:
            ax.scatter(
                embeddings_2d[rep_indices, 0],
                embeddings_2d[rep_indices, 1],
                c=self.config.representative_color,
                marker="*",
                s=(self.config.point_size + 6) ** 2,
                edgecolors="black",
                linewidths=0.5,
                label="Representatives",
                zorder=10
            )

        # Customize plot
        ax.set_xlabel(f"{self.config.reduction_method.upper()} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{self.config.reduction_method.upper()} Dimension 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Legend (limit entries if too many clusters)
        if self.config.show_legend and n_clusters <= 15:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=9
            )
        elif self.config.show_legend:
            # Create summary legend
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                       markersize=8, label=f"{n_clusters} Clusters"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                       markeredgecolor=self.config.edge_case_color, markersize=8,
                       markeredgewidth=2, label=f"{len(results.edge_case_indices)} Edge Cases"),
                Line2D([0], [0], marker="D", color="w",
                       markerfacecolor=self.config.void_cluster_color,
                       markersize=8, label=f"{len(results.void_cluster_ids)} Void Clusters"),
                Line2D([0], [0], marker="*", color="w",
                       markerfacecolor=self.config.representative_color,
                       markeredgecolor="black", markersize=10, label="Representatives"),
            ]
            ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.02, 0.5))

        plt.tight_layout()

        # Save if requested
        if save_path is None and self.config.save_png:
            save_path = Path(self.config.output_dir) / "embedding_plot.png"

        if save_path:
            fig.savefig(str(save_path), dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved static plot to {save_path}")

        return fig

    def get_embeddings_2d(self) -> Optional[NDArray[np.float32]]:
        """Get the 2D embeddings if already computed."""
        return self._embeddings_2d
