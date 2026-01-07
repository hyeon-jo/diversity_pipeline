"""
Streamlit dashboard for interactive exploration of clustering results.

Provides a unified interface for:
- Embedding space visualization
- Cluster gallery browsing
- Distribution analysis
- Network graph exploration
- Statistics overview
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from .config import VisualizerConfig

if TYPE_CHECKING:
    from video_curation_pipeline import PipelineResults

logger = logging.getLogger(__name__)


def create_dashboard_app(
    results: "PipelineResults",
    frame_base_dir: Optional[Union[str, Path]] = None,
    config: Optional[VisualizerConfig] = None,
    video_names: Optional[list[str]] = None
):
    """
    Create a Streamlit dashboard app.

    This function returns the dashboard code that can be run with Streamlit.
    Save it to a file and run with: streamlit run dashboard_app.py

    Args:
        results: Pipeline results.
        frame_base_dir: Base directory for frame files.
        config: Visualization configuration.
        video_names: Optional list of video names.
    """
    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "streamlit required for dashboard. "
            "Install with: pip install streamlit"
        )

    from .embedding_plot import EmbeddingPlotter
    from .cluster_gallery import ClusterGallery
    from .distribution_chart import DistributionChart
    from .network_graph import NetworkGraphPlotter

    config = config or VisualizerConfig()

    # Page config
    st.set_page_config(
        page_title="Video Curation Dashboard",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Video Curation Pipeline Dashboard")

    # Sidebar - Configuration
    st.sidebar.header("⚙️ Settings")

    reduction_method = st.sidebar.selectbox(
        "Dimension Reduction",
        ["umap", "tsne"],
        index=0 if config.reduction_method == "umap" else 1
    )

    k_neighbors = st.sidebar.slider(
        "k-NN Neighbors",
        min_value=5,
        max_value=50,
        value=config.n_neighbors
    )

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🗺️ Embedding Space",
        "🖼️ Cluster Gallery",
        "📈 Distribution",
        "🔗 Network"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("Clustering Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Samples", len(results.cluster_labels))
        with col2:
            st.metric("Clusters", len(results.cluster_info))
        with col3:
            st.metric("Edge Cases", len(results.edge_case_indices))
        with col4:
            st.metric("Void Clusters", len(results.void_cluster_ids))

        # Additional stats
        st.subheader("Cluster Statistics")

        dist_chart = DistributionChart(config)
        stats = dist_chart.get_statistics_dict(results)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Cluster Size", f"{stats['cluster_size_mean']:.1f}")
            st.metric("Min Size", stats['cluster_size_min'])
        with col2:
            st.metric("Median Size", f"{stats['cluster_size_median']:.1f}")
            st.metric("Max Size", stats['cluster_size_max'])
        with col3:
            st.metric("Size Std Dev", f"{stats['cluster_size_std']:.1f}")
            st.metric("Micro-clusters", stats['n_micro_clusters'])

        # Cluster table
        st.subheader("Cluster Details")

        cluster_data = []
        for cid, info in sorted(results.cluster_info.items()):
            cluster_data.append({
                "Cluster ID": cid,
                "Size": info.size,
                "Density": f"{info.density:.4f}",
                "Type": "Micro" if info.is_micro_cluster else ("Void" if info.is_low_density else "Normal"),
                "Caption": results.captions.get(cid, "N/A")[:50] + "..." if results.captions.get(cid, "") else "N/A"
            })

        st.dataframe(cluster_data, use_container_width=True)

    # Tab 2: Embedding Space
    with tab2:
        st.header("Embedding Space Visualization")

        plotter = EmbeddingPlotter(config)
        config.reduction_method = reduction_method

        with st.spinner(f"Computing {reduction_method.upper()} projection..."):
            fig = plotter.plot_interactive(
                results,
                video_names=video_names,
                title=f"Video Embeddings ({reduction_method.upper()})"
            )

        st.plotly_chart(fig, use_container_width=True)

        # Show 2D coordinates
        if st.checkbox("Show 2D Coordinates"):
            embeddings_2d = plotter.get_embeddings_2d()
            if embeddings_2d is not None:
                coord_data = []
                for i, (x, y) in enumerate(embeddings_2d):
                    name = video_names[i] if video_names else f"Video_{i}"
                    coord_data.append({
                        "Video": name,
                        "X": f"{x:.4f}",
                        "Y": f"{y:.4f}",
                        "Cluster": results.cluster_labels[i]
                    })
                st.dataframe(coord_data, use_container_width=True, height=300)

    # Tab 3: Cluster Gallery
    with tab3:
        st.header("Cluster Gallery")

        if frame_base_dir:
            gallery = ClusterGallery(config)

            # Cluster selector
            selected_cluster = st.selectbox(
                "Select Cluster",
                options=["All"] + sorted(results.cluster_info.keys()),
                format_func=lambda x: f"Cluster {x}" if x != "All" else "All Clusters"
            )

            if selected_cluster == "All":
                # Show gallery grid
                fig = gallery.create_gallery(
                    results,
                    frame_base_dir,
                    title="All Clusters"
                )
                st.pyplot(fig)
            else:
                # Show single cluster details
                info = results.cluster_info[selected_cluster]

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader(f"Cluster {selected_cluster}")
                    st.write(f"**Size:** {info.size} samples")
                    st.write(f"**Density:** {info.density:.4f}")

                    if info.is_micro_cluster:
                        st.warning("⚠️ Micro-cluster (potential edge case)")
                    if info.is_low_density:
                        st.info("🔍 Low density (void cluster)")

                    if selected_cluster in results.captions:
                        st.write("**Caption:**")
                        st.write(results.captions[selected_cluster])

                with col2:
                    # Show representative frame
                    if info.representative_path:
                        frame = gallery._load_representative_frame(
                            info.representative_path,
                            Path(frame_base_dir)
                        )
                        if frame:
                            st.image(frame, caption=f"Representative: {info.representative_path}")
        else:
            st.warning("Frame base directory not provided. Gallery view unavailable.")

    # Tab 4: Distribution
    with tab4:
        st.header("Cluster Distribution")

        dist_chart = DistributionChart(config)

        chart_type = st.radio(
            "Chart Type",
            ["Sunburst", "Treemap", "Bar Chart"],
            horizontal=True
        )

        if chart_type == "Sunburst":
            fig = dist_chart.plot_sunburst(results)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Treemap":
            fig = dist_chart.plot_treemap(results)
            st.plotly_chart(fig, use_container_width=True)
        else:
            top_n = st.slider("Show Top N Clusters", 10, 50, 30)
            fig = dist_chart.plot_bar_chart(results, top_n=top_n)
            st.pyplot(fig)

        # Statistics summary
        st.subheader("Statistics Summary")
        fig_stats = dist_chart.plot_statistics_summary(results)
        st.pyplot(fig_stats)

    # Tab 5: Network
    with tab5:
        st.header("k-NN Network Graph")

        network_plotter = NetworkGraphPlotter(config)

        with st.spinner(f"Building k-NN graph (k={k_neighbors})..."):
            network_plotter.build_knn_graph(results.embeddings, k=k_neighbors)

        # Show metrics
        metrics = network_plotter.get_graph_metrics()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nodes", metrics['n_nodes'])
        with col2:
            st.metric("Edges", metrics['n_edges'])
        with col3:
            st.metric("Avg Degree", f"{metrics['avg_degree']:.1f}")
        with col4:
            st.metric("Clustering Coef", f"{metrics['avg_clustering']:.3f}")

        # Static plot
        st.subheader("Network Visualization")
        fig = network_plotter.plot_static(results, video_names=video_names)
        st.pyplot(fig)

        st.info("💡 For interactive network exploration, use the HTML export option.")


def generate_dashboard_script(
    results_path: str,
    frame_base_dir: Optional[str] = None,
    output_path: str = "dashboard_app.py"
) -> str:
    """
    Generate a standalone Streamlit dashboard script.

    Args:
        results_path: Path to saved pipeline results (pickle file).
        frame_base_dir: Base directory for frame files.
        output_path: Path to save the dashboard script.

    Returns:
        Path to the generated script.
    """
    frame_dir_line = f'FRAME_BASE_DIR = "{frame_base_dir}"' if frame_base_dir else 'FRAME_BASE_DIR = None'

    script = f'''#!/usr/bin/env python3
"""
Video Curation Pipeline Dashboard

Run with: streamlit run {output_path}
"""

import pickle
from pathlib import Path

import streamlit as st

# Configuration
RESULTS_PATH = "{results_path}"
{frame_dir_line}

# Load results
@st.cache_data
def load_results():
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)

# Import visualizer components
import sys
sys.path.insert(0, str(Path(__file__).parent))

from visualizer import VisualizerConfig
from visualizer.dashboard import create_dashboard_app

# Run dashboard
results = load_results()
config = VisualizerConfig()
create_dashboard_app(results, frame_base_dir=FRAME_BASE_DIR, config=config)
'''

    Path(output_path).write_text(script)
    logger.info(f"Generated dashboard script: {output_path}")
    logger.info(f"Run with: streamlit run {output_path}")

    return output_path
