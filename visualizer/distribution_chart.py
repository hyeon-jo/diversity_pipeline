"""
Distribution visualization using Sunburst and Treemap charts.

Shows cluster size distribution and hierarchical relationships with:
- Interactive Plotly Sunburst charts
- Treemap views for size comparison
- Pie/bar charts for simple distribution views
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from .config import VisualizerConfig

if TYPE_CHECKING:
    from video_curation_pipeline import PipelineResults

logger = logging.getLogger(__name__)


class DistributionChart:
    """
    Creates distribution visualizations for cluster analysis.

    Provides multiple views:
    - Sunburst: Hierarchical view of cluster sizes
    - Treemap: Area-based size comparison
    - Bar chart: Simple size ranking
    - Statistics summary
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize the distribution chart plotter.

        Args:
            config: Visualization configuration.
        """
        self.config = config or VisualizerConfig()

    def plot_sunburst(
        self,
        results: "PipelineResults",
        title: str = "Cluster Size Distribution",
        save_path: Optional[str] = None
    ) -> "plotly.graph_objects.Figure":
        """
        Create an interactive Sunburst chart.

        The sunburst shows:
        - Center: Total dataset
        - Inner ring: Cluster categories (Normal, Void, Edge)
        - Outer ring: Individual clusters

        Args:
            results: Pipeline results.
            title: Chart title.
            save_path: Path to save HTML file.

        Returns:
            Plotly Figure object.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly required for Sunburst charts. "
                "Install with: pip install plotly"
            )

        # Prepare data
        labels = ["Total"]
        parents = [""]
        values = [len(results.cluster_labels)]
        colors = ["#FFFFFF"]

        # Add category level
        normal_size = 0
        void_size = 0
        edge_size = 0

        for cluster_id, info in results.cluster_info.items():
            if info.is_micro_cluster:
                edge_size += info.size
            elif info.is_low_density:
                void_size += info.size
            else:
                normal_size += info.size

        if normal_size > 0:
            labels.append("Normal Clusters")
            parents.append("Total")
            values.append(normal_size)
            colors.append("#90EE90")  # Light green

        if void_size > 0:
            labels.append("Void Clusters")
            parents.append("Total")
            values.append(void_size)
            colors.append(self.config.void_cluster_color)

        if edge_size > 0:
            labels.append("Micro Clusters")
            parents.append("Total")
            values.append(edge_size)
            colors.append(self.config.edge_case_color)

        # Add individual clusters
        for cluster_id, info in sorted(
            results.cluster_info.items(),
            key=lambda x: x[1].size,
            reverse=True
        ):
            cluster_label = f"Cluster {cluster_id}"
            labels.append(cluster_label)
            values.append(info.size)

            if info.is_micro_cluster:
                parents.append("Micro Clusters")
                colors.append("#FF9999")  # Light red
            elif info.is_low_density:
                parents.append("Void Clusters")
                colors.append("#7FDBDB")  # Light teal
            else:
                parents.append("Normal Clusters")
                colors.append("#98FB98")  # Pale green

        # Create sunburst
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors),
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Size: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5),
            width=800,
            height=800
        )

        # Save if requested
        if save_path is None and self.config.save_html:
            save_path = Path(self.config.output_dir) / "distribution_sunburst.html"

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Saved Sunburst chart to {save_path}")

        return fig

    def plot_treemap(
        self,
        results: "PipelineResults",
        title: str = "Cluster Size Treemap",
        save_path: Optional[str] = None
    ) -> "plotly.graph_objects.Figure":
        """
        Create an interactive Treemap chart.

        Args:
            results: Pipeline results.
            title: Chart title.
            save_path: Path to save HTML file.

        Returns:
            Plotly Figure object.
        """
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError(
                "plotly required for Treemap charts. "
                "Install with: pip install plotly"
            )

        # Prepare data for treemap
        data = []

        for cluster_id, info in results.cluster_info.items():
            category = "Normal"
            if info.is_micro_cluster:
                category = "Micro-cluster"
            elif info.is_low_density:
                category = "Void"

            # Get caption if available
            caption = results.captions.get(cluster_id, f"Cluster {cluster_id}")
            if len(caption) > 40:
                caption = caption[:37] + "..."

            data.append({
                "Cluster": f"C{cluster_id}",
                "Category": category,
                "Size": info.size,
                "Density": info.density,
                "Caption": caption
            })

        import pandas as pd
        df = pd.DataFrame(data)

        # Create treemap
        fig = px.treemap(
            df,
            path=["Category", "Cluster"],
            values="Size",
            color="Category",
            color_discrete_map={
                "Normal": "#90EE90",
                "Void": self.config.void_cluster_color,
                "Micro-cluster": self.config.edge_case_color
            },
            hover_data=["Size", "Density", "Caption"],
            title=title
        )

        fig.update_layout(
            width=1000,
            height=700
        )

        fig.update_traces(
            textinfo="label+value",
            hovertemplate="<b>%{label}</b><br>Size: %{value}<br>%{customdata}<extra></extra>"
        )

        # Save if requested
        if save_path is None and self.config.save_html:
            save_path = Path(self.config.output_dir) / "distribution_treemap.html"

        if save_path:
            fig.write_html(str(save_path))
            logger.info(f"Saved Treemap chart to {save_path}")

        return fig

    def plot_bar_chart(
        self,
        results: "PipelineResults",
        title: str = "Cluster Size Distribution",
        top_n: int = 30,
        save_path: Optional[str] = None
    ) -> "matplotlib.figure.Figure":
        """
        Create a horizontal bar chart of cluster sizes.

        Args:
            results: Pipeline results.
            title: Chart title.
            top_n: Show only top N clusters.
            save_path: Path to save image.

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        # Sort clusters by size
        sorted_clusters = sorted(
            results.cluster_info.items(),
            key=lambda x: x[1].size,
            reverse=True
        )[:top_n]

        cluster_ids = [f"C{cid}" for cid, _ in sorted_clusters]
        sizes = [info.size for _, info in sorted_clusters]

        # Determine colors
        colors = []
        for cid, info in sorted_clusters:
            if info.is_micro_cluster:
                colors.append(self.config.edge_case_color)
            elif info.is_low_density:
                colors.append(self.config.void_cluster_color)
            else:
                colors.append("#90EE90")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(6, len(cluster_ids) * 0.3)))

        y_pos = np.arange(len(cluster_ids))
        bars = ax.barh(y_pos, sizes, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cluster_ids)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Samples")
        ax.set_title(title, fontweight="bold")

        # Add value labels
        for bar, size in zip(bars, sizes):
            width = bar.get_width()
            ax.text(
                width + max(sizes) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{size}",
                va="center",
                fontsize=8
            )

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#90EE90", edgecolor="black", label="Normal"),
            Patch(facecolor=self.config.void_cluster_color, edgecolor="black", label="Void"),
            Patch(facecolor=self.config.edge_case_color, edgecolor="black", label="Micro-cluster"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()

        # Save if requested
        if save_path is None and self.config.save_png:
            save_path = Path(self.config.output_dir) / "distribution_bar.png"

        if save_path:
            fig.savefig(str(save_path), dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved bar chart to {save_path}")

        return fig

    def plot_statistics_summary(
        self,
        results: "PipelineResults",
        save_path: Optional[str] = None
    ) -> "matplotlib.figure.Figure":
        """
        Create a statistics summary dashboard.

        Args:
            results: Pipeline results.
            save_path: Path to save image.

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig = plt.figure(figsize=(14, 8))

        # Compute statistics
        total_samples = len(results.cluster_labels)
        n_clusters = len(results.cluster_info)
        sizes = [info.size for info in results.cluster_info.values()]
        densities = [info.density for info in results.cluster_info.values()]

        n_normal = sum(1 for info in results.cluster_info.values()
                       if not info.is_micro_cluster and not info.is_low_density)
        n_void = len(results.void_cluster_ids)
        n_micro = sum(1 for info in results.cluster_info.values() if info.is_micro_cluster)

        # Layout: 2 rows, 3 columns
        # Top row: Key metrics
        # Bottom row: Distribution plots

        # Top row - Key metrics cards
        metrics = [
            ("Total Samples", f"{total_samples:,}", "#3498db"),
            ("Clusters", f"{n_clusters}", "#2ecc71"),
            ("Edge Cases", f"{len(results.edge_case_indices)}", self.config.edge_case_color),
            ("Void Clusters", f"{n_void}", self.config.void_cluster_color),
            ("Avg Cluster Size", f"{np.mean(sizes):.1f}", "#9b59b6"),
            ("Median Size", f"{np.median(sizes):.1f}", "#e74c3c"),
        ]

        for i, (label, value, color) in enumerate(metrics):
            ax = fig.add_subplot(2, 6, i + 1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            # Draw card background
            card = FancyBboxPatch(
                (0.05, 0.1), 0.9, 0.8,
                boxstyle="round,pad=0.05,rounding_size=0.1",
                facecolor=color,
                alpha=0.2,
                edgecolor=color,
                linewidth=2
            )
            ax.add_patch(card)

            ax.text(0.5, 0.65, value, ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
            ax.text(0.5, 0.3, label, ha="center", va="center",
                    fontsize=10, color="#333")

        # Bottom left: Cluster type pie chart
        ax_pie = fig.add_subplot(2, 3, 4)
        type_sizes = [n_normal, n_void, n_micro]
        type_labels = ["Normal", "Void", "Micro"]
        type_colors = ["#90EE90", self.config.void_cluster_color, self.config.edge_case_color]

        # Filter out zeros
        non_zero = [(s, l, c) for s, l, c in zip(type_sizes, type_labels, type_colors) if s > 0]
        if non_zero:
            sizes_nz, labels_nz, colors_nz = zip(*non_zero)
            ax_pie.pie(
                sizes_nz,
                labels=labels_nz,
                colors=colors_nz,
                autopct="%1.1f%%",
                startangle=90
            )
            ax_pie.set_title("Cluster Types", fontweight="bold")

        # Bottom middle: Size distribution histogram
        ax_hist = fig.add_subplot(2, 3, 5)
        ax_hist.hist(sizes, bins=min(20, n_clusters), color="#3498db", edgecolor="white")
        ax_hist.set_xlabel("Cluster Size")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Size Distribution", fontweight="bold")
        ax_hist.axvline(np.mean(sizes), color="red", linestyle="--", label=f"Mean: {np.mean(sizes):.1f}")
        ax_hist.axvline(np.median(sizes), color="orange", linestyle="--", label=f"Median: {np.median(sizes):.1f}")
        ax_hist.legend(fontsize=8)

        # Bottom right: Density distribution
        ax_density = fig.add_subplot(2, 3, 6)
        ax_density.hist(densities, bins=min(20, n_clusters), color="#2ecc71", edgecolor="white")
        ax_density.set_xlabel("Cluster Density")
        ax_density.set_ylabel("Count")
        ax_density.set_title("Density Distribution", fontweight="bold")

        plt.suptitle("Clustering Statistics Summary", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        # Save if requested
        if save_path is None and self.config.save_png:
            save_path = Path(self.config.output_dir) / "statistics_summary.png"

        if save_path:
            fig.savefig(str(save_path), dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved statistics summary to {save_path}")

        return fig

    def get_statistics_dict(self, results: "PipelineResults") -> dict:
        """
        Get clustering statistics as a dictionary.

        Args:
            results: Pipeline results.

        Returns:
            Dictionary of statistics.
        """
        sizes = [info.size for info in results.cluster_info.values()]
        densities = [info.density for info in results.cluster_info.values()]

        return {
            "total_samples": len(results.cluster_labels),
            "n_clusters": len(results.cluster_info),
            "n_edge_cases": len(results.edge_case_indices),
            "n_void_clusters": len(results.void_cluster_ids),
            "n_micro_clusters": sum(1 for info in results.cluster_info.values() if info.is_micro_cluster),
            "cluster_size_mean": float(np.mean(sizes)),
            "cluster_size_std": float(np.std(sizes)),
            "cluster_size_median": float(np.median(sizes)),
            "cluster_size_min": int(np.min(sizes)),
            "cluster_size_max": int(np.max(sizes)),
            "density_mean": float(np.mean(densities)),
            "density_std": float(np.std(densities)),
        }
