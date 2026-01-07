"""
Video Curation Pipeline Visualizer

A comprehensive visualization module for analyzing clustering results from
the video curation pipeline. Provides multiple visualization types:

- EmbeddingPlot: t-SNE/UMAP 2D scatter plots of embeddings
- ClusterGallery: Grid display of representative video frames
- DistributionChart: Sunburst/Treemap for cluster size distribution
- NetworkGraph: k-NN graph visualization
- Dashboard: Streamlit-based interactive dashboard

Usage:
    from visualizer import Visualizer, VisualizerConfig

    viz = Visualizer(config=VisualizerConfig(output_dir="viz_output"))
    viz.plot_embeddings(results)
    viz.create_gallery(results, frame_base_dir="./trainlake")
    viz.plot_distribution(results)
    viz.plot_network(results)
    viz.launch_dashboard(results)
"""

from .config import VisualizerConfig
from .embedding_plot import EmbeddingPlotter
from .cluster_gallery import ClusterGallery
from .distribution_chart import DistributionChart
from .network_graph import NetworkGraphPlotter
from .main import Visualizer

__all__ = [
    "Visualizer",
    "VisualizerConfig",
    "EmbeddingPlotter",
    "ClusterGallery",
    "DistributionChart",
    "NetworkGraphPlotter",
]

__version__ = "1.0.0"
