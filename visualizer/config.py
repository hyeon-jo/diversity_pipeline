"""
Configuration for the visualizer module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VisualizerConfig:
    """Configuration for visualization outputs."""

    # Output settings
    output_dir: str = "viz_output"
    save_html: bool = True
    save_png: bool = True
    dpi: int = 150

    # Embedding plot settings
    reduction_method: str = "umap"  # "umap" or "tsne"
    n_neighbors: int = 15  # For UMAP
    min_dist: float = 0.1  # For UMAP
    perplexity: int = 30  # For t-SNE
    point_size: int = 8
    show_legend: bool = True

    # Color scheme
    colormap: str = "tab20"  # Matplotlib colormap for clusters
    edge_case_color: str = "#FF6B6B"  # Red for edge cases
    void_cluster_color: str = "#4ECDC4"  # Teal for void clusters
    representative_color: str = "#FFE66D"  # Yellow for representatives

    # Gallery settings
    gallery_cols: int = 5
    thumbnail_size: tuple[int, int] = field(default_factory=lambda: (224, 224))
    show_captions: bool = True
    max_clusters_per_page: int = 20

    # Network graph settings
    network_layout: str = "spring"  # "spring", "kamada_kawai", "circular"
    edge_alpha: float = 0.3
    node_size_scale: float = 50.0

    # Dashboard settings
    dashboard_port: int = 8501
    dashboard_theme: str = "light"

    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
