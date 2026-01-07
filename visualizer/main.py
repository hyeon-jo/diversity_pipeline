"""
Main Visualizer class that integrates all visualization components.

Provides a unified interface for generating all visualizations:
- Embedding plots (t-SNE/UMAP)
- Cluster galleries
- Distribution charts
- Network graphs
- Dashboard
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .config import VisualizerConfig
from .embedding_plot import EmbeddingPlotter
from .cluster_gallery import ClusterGallery
from .distribution_chart import DistributionChart
from .network_graph import NetworkGraphPlotter

if TYPE_CHECKING:
    from video_curation_pipeline import PipelineResults

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Unified interface for all visualization components.

    Example usage:
        from visualizer import Visualizer, VisualizerConfig

        config = VisualizerConfig(output_dir="viz_output", reduction_method="umap")
        viz = Visualizer(config)

        # Generate all visualizations
        viz.generate_all(results, frame_base_dir="./trainlake")

        # Or generate individually
        viz.plot_embeddings(results)
        viz.create_gallery(results, frame_base_dir="./trainlake")
        viz.plot_distribution(results)
        viz.plot_network(results)
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize the Visualizer.

        Args:
            config: Visualization configuration. Uses defaults if None.
        """
        self.config = config or VisualizerConfig()

        # Initialize component plotters
        self.embedding_plotter = EmbeddingPlotter(self.config)
        self.cluster_gallery = ClusterGallery(self.config)
        self.distribution_chart = DistributionChart(self.config)
        self.network_plotter = NetworkGraphPlotter(self.config)

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Visualizer initialized with output_dir: {self.config.output_dir}")

    def plot_embeddings(
        self,
        results: "PipelineResults",
        video_names: Optional[list[str]] = None,
        interactive: bool = True,
        static: bool = True,
        title: str = "Video Embedding Space"
    ) -> dict[str, Path]:
        """
        Generate embedding space visualizations.

        Args:
            results: Pipeline results.
            video_names: Optional list of video names.
            interactive: Generate interactive HTML plot.
            static: Generate static PNG plot.
            title: Plot title.

        Returns:
            Dictionary of output file paths.
        """
        outputs = {}

        if interactive:
            html_path = Path(self.config.output_dir) / "embedding_plot.html"
            self.embedding_plotter.plot_interactive(
                results,
                video_names=video_names,
                title=title,
                save_path=str(html_path)
            )
            outputs["interactive"] = html_path

        if static:
            png_path = Path(self.config.output_dir) / "embedding_plot.png"
            self.embedding_plotter.plot_static(
                results,
                video_names=video_names,
                title=title,
                save_path=str(png_path)
            )
            outputs["static"] = png_path

        return outputs

    def create_gallery(
        self,
        results: "PipelineResults",
        frame_base_dir: Union[str, Path],
        html: bool = True,
        static: bool = True,
        title: str = "Cluster Gallery"
    ) -> dict[str, Path]:
        """
        Generate cluster gallery visualizations.

        Args:
            results: Pipeline results.
            frame_base_dir: Base directory containing frame directories.
            html: Generate interactive HTML gallery.
            static: Generate static PNG gallery.
            title: Gallery title.

        Returns:
            Dictionary of output file paths.
        """
        outputs = {}

        if html:
            html_path = Path(self.config.output_dir) / "cluster_gallery.html"
            self.cluster_gallery.create_html_gallery(
                results,
                frame_base_dir,
                title=title,
                save_path=str(html_path)
            )
            outputs["html"] = html_path

        if static:
            png_path = Path(self.config.output_dir) / "cluster_gallery.png"
            self.cluster_gallery.create_gallery(
                results,
                frame_base_dir,
                title=title,
                save_path=str(png_path)
            )
            outputs["static"] = png_path

        return outputs

    def plot_distribution(
        self,
        results: "PipelineResults",
        sunburst: bool = True,
        treemap: bool = True,
        bar_chart: bool = True,
        statistics: bool = True
    ) -> dict[str, Path]:
        """
        Generate distribution visualizations.

        Args:
            results: Pipeline results.
            sunburst: Generate Sunburst chart.
            treemap: Generate Treemap chart.
            bar_chart: Generate bar chart.
            statistics: Generate statistics summary.

        Returns:
            Dictionary of output file paths.
        """
        outputs = {}

        if sunburst:
            path = Path(self.config.output_dir) / "distribution_sunburst.html"
            self.distribution_chart.plot_sunburst(results, save_path=str(path))
            outputs["sunburst"] = path

        if treemap:
            path = Path(self.config.output_dir) / "distribution_treemap.html"
            self.distribution_chart.plot_treemap(results, save_path=str(path))
            outputs["treemap"] = path

        if bar_chart:
            path = Path(self.config.output_dir) / "distribution_bar.png"
            self.distribution_chart.plot_bar_chart(results, save_path=str(path))
            outputs["bar_chart"] = path

        if statistics:
            path = Path(self.config.output_dir) / "statistics_summary.png"
            self.distribution_chart.plot_statistics_summary(results, save_path=str(path))
            outputs["statistics"] = path

        return outputs

    def plot_network(
        self,
        results: "PipelineResults",
        video_names: Optional[list[str]] = None,
        k: int = 15,
        interactive: bool = True,
        static: bool = True,
        title: str = "k-NN Similarity Network"
    ) -> dict[str, Path]:
        """
        Generate network graph visualizations.

        Args:
            results: Pipeline results.
            video_names: Optional list of video names.
            k: Number of nearest neighbors.
            interactive: Generate interactive HTML network.
            static: Generate static PNG network.
            title: Network title.

        Returns:
            Dictionary of output file paths.
        """
        outputs = {}

        # Build graph first
        self.network_plotter.build_knn_graph(results.embeddings, k=k)

        if interactive:
            html_path = Path(self.config.output_dir) / "network_graph.html"
            self.network_plotter.plot_interactive(
                results,
                video_names=video_names,
                k=k,
                title=title,
                save_path=str(html_path)
            )
            outputs["interactive"] = html_path

        if static:
            png_path = Path(self.config.output_dir) / "network_graph.png"
            self.network_plotter.plot_static(
                results,
                video_names=video_names,
                k=k,
                title=title,
                save_path=str(png_path)
            )
            outputs["static"] = png_path

        return outputs

    def generate_all(
        self,
        results: "PipelineResults",
        frame_base_dir: Optional[Union[str, Path]] = None,
        video_names: Optional[list[str]] = None,
        k_neighbors: int = 15
    ) -> dict[str, dict[str, Path]]:
        """
        Generate all visualizations.

        Args:
            results: Pipeline results.
            frame_base_dir: Base directory for frame files (required for gallery).
            video_names: Optional list of video names.
            k_neighbors: Number of neighbors for k-NN graph.

        Returns:
            Nested dictionary of all output file paths.
        """
        logger.info("Generating all visualizations...")

        all_outputs = {}

        # Embedding plots
        logger.info("1/4 Creating embedding plots...")
        all_outputs["embeddings"] = self.plot_embeddings(
            results,
            video_names=video_names
        )

        # Distribution charts
        logger.info("2/4 Creating distribution charts...")
        all_outputs["distribution"] = self.plot_distribution(results)

        # Network graphs
        logger.info("3/4 Creating network graphs...")
        all_outputs["network"] = self.plot_network(
            results,
            video_names=video_names,
            k=k_neighbors
        )

        # Cluster gallery (requires frame_base_dir)
        if frame_base_dir:
            logger.info("4/4 Creating cluster gallery...")
            all_outputs["gallery"] = self.create_gallery(
                results,
                frame_base_dir
            )
        else:
            logger.warning("Skipping gallery: frame_base_dir not provided")

        logger.info(f"All visualizations saved to: {self.config.output_dir}")

        return all_outputs

    def generate_report(
        self,
        results: "PipelineResults",
        frame_base_dir: Optional[Union[str, Path]] = None,
        video_names: Optional[list[str]] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate an HTML report combining all visualizations.

        Args:
            results: Pipeline results.
            frame_base_dir: Base directory for frame files.
            video_names: Optional list of video names.
            output_path: Path to save the report.

        Returns:
            Path to the generated report.
        """
        # Generate all visualizations first
        all_outputs = self.generate_all(
            results,
            frame_base_dir=frame_base_dir,
            video_names=video_names
        )

        # Get statistics
        stats = self.distribution_chart.get_statistics_dict(results)

        # Build HTML report
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Video Curation Pipeline Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; text-align: center; }}
        h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .viz-link {{
            display: inline-block;
            padding: 10px 20px;
            background: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 5px;
        }}
        .viz-link:hover {{ background: #2980b9; }}
        img {{ max-width: 100%; height: auto; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🎬 Video Curation Pipeline Report</h1>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value">{stats['total_samples']:,}</div>
            <div class="stat-label">Total Samples</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_clusters']}</div>
            <div class="stat-label">Clusters</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_edge_cases']}</div>
            <div class="stat-label">Edge Cases</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_void_clusters']}</div>
            <div class="stat-label">Void Clusters</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['cluster_size_mean']:.1f}</div>
            <div class="stat-label">Avg Cluster Size</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['n_micro_clusters']}</div>
            <div class="stat-label">Micro-clusters</div>
        </div>
    </div>

    <div class="section">
        <h2>🗺️ Embedding Space</h2>
        <p>Interactive visualization of video embeddings reduced to 2D using {self.config.reduction_method.upper()}.</p>
        <a class="viz-link" href="embedding_plot.html" target="_blank">Open Interactive Plot</a>
        <br><br>
        <img src="embedding_plot.png" alt="Embedding Plot">
    </div>

    <div class="section">
        <h2>📊 Cluster Distribution</h2>
        <p>Size distribution of discovered clusters.</p>
        <a class="viz-link" href="distribution_sunburst.html" target="_blank">Sunburst Chart</a>
        <a class="viz-link" href="distribution_treemap.html" target="_blank">Treemap</a>
        <br><br>
        <img src="distribution_bar.png" alt="Distribution Bar Chart">
        <br><br>
        <img src="statistics_summary.png" alt="Statistics Summary">
    </div>

    <div class="section">
        <h2>🔗 Similarity Network</h2>
        <p>k-NN graph showing similarity relationships between videos.</p>
        <a class="viz-link" href="network_graph.html" target="_blank">Open Interactive Network</a>
        <br><br>
        <img src="network_graph.png" alt="Network Graph">
    </div>
"""

        if "gallery" in all_outputs:
            html += """
    <div class="section">
        <h2>🖼️ Cluster Gallery</h2>
        <p>Representative frames from each cluster.</p>
        <a class="viz-link" href="cluster_gallery.html" target="_blank">Open Interactive Gallery</a>
        <br><br>
        <img src="cluster_gallery.png" alt="Cluster Gallery">
    </div>
"""

        html += """
    <footer style="text-align: center; color: #999; margin-top: 40px;">
        Generated by Video Curation Pipeline Visualizer
    </footer>
</body>
</html>
"""

        # Save report
        if output_path is None:
            output_path = Path(self.config.output_dir) / "report.html"

        Path(output_path).write_text(html)
        logger.info(f"Generated report: {output_path}")

        return str(output_path)

    def launch_dashboard(
        self,
        results: "PipelineResults",
        frame_base_dir: Optional[Union[str, Path]] = None,
        video_names: Optional[list[str]] = None,
        port: int = 8501
    ) -> None:
        """
        Launch the interactive Streamlit dashboard.

        Args:
            results: Pipeline results.
            frame_base_dir: Base directory for frame files.
            video_names: Optional list of video names.
            port: Port for the Streamlit server.
        """
        from .dashboard import create_dashboard_app

        logger.info(f"Launching dashboard on port {port}...")
        create_dashboard_app(
            results,
            frame_base_dir=frame_base_dir,
            config=self.config,
            video_names=video_names
        )
