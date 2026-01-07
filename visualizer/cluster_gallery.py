"""
Cluster gallery visualization showing representative video frames.

Creates grid displays of:
- Representative frames for each cluster
- Cluster statistics and captions
- Edge case / void cluster highlighting
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


class ClusterGallery:
    """
    Creates visual galleries of cluster representatives.

    Displays thumbnail grids with cluster information, captions,
    and visual indicators for edge cases and void clusters.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize the cluster gallery.

        Args:
            config: Visualization configuration.
        """
        self.config = config or VisualizerConfig()

    def _load_representative_frame(
        self,
        video_path: str,
        frame_base_dir: Path,
        frame_index: int = 0
    ) -> Optional["PIL.Image.Image"]:
        """
        Load a representative frame from a video's frame directory.

        Args:
            video_path: Video name or path.
            frame_base_dir: Base directory containing frame directories.
            frame_index: Which frame to load (0 = first sampled frame).

        Returns:
            PIL Image or None if not found.
        """
        from PIL import Image

        video_name = Path(video_path).stem
        if video_name.endswith(".mp4"):
            video_name = video_name[:-4]

        # Search for frame directory
        for dir_path in frame_base_dir.rglob("*"):
            if dir_path.is_dir() and dir_path.name == video_name:
                # Look for CMR_GT_Frame subdirectory
                frame_dir = dir_path / "RAW_DB"
                if frame_dir.exists():
                    cmr_dirs = list(frame_dir.glob("*_CMR*/CMR_GT_Frame"))
                    if cmr_dirs:
                        frame_files = sorted(cmr_dirs[0].glob("*.jpg"))
                        if not frame_files:
                            frame_files = sorted(cmr_dirs[0].glob("*.JPG"))

                        if frame_files:
                            # Get middle frame for better representation
                            mid_idx = len(frame_files) // 2
                            return Image.open(frame_files[mid_idx]).convert("RGB")

        return None

    def _create_placeholder_image(
        self,
        text: str = "No Frame",
        size: tuple[int, int] = (224, 224)
    ) -> "PIL.Image.Image":
        """Create a placeholder image with text."""
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", size, color=(128, 128, 128))
        draw = ImageDraw.Draw(img)

        # Try to use a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Center text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        return img

    def create_gallery(
        self,
        results: "PipelineResults",
        frame_base_dir: Union[str, Path],
        title: str = "Cluster Gallery",
        save_path: Optional[str] = None,
        sort_by: str = "size"  # "size", "id", "density"
    ) -> "matplotlib.figure.Figure":
        """
        Create a gallery grid of cluster representatives.

        Args:
            results: Pipeline results containing cluster info.
            frame_base_dir: Base directory containing frame directories.
            title: Gallery title.
            save_path: Path to save the image.
            sort_by: How to sort clusters ("size", "id", "density").

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image

        frame_base_dir = Path(frame_base_dir)

        # Sort clusters
        cluster_ids = list(results.cluster_info.keys())
        if sort_by == "size":
            cluster_ids.sort(key=lambda x: results.cluster_info[x].size, reverse=True)
        elif sort_by == "density":
            cluster_ids.sort(key=lambda x: results.cluster_info[x].density, reverse=True)
        else:
            cluster_ids.sort()

        # Limit to max clusters per page
        if len(cluster_ids) > self.config.max_clusters_per_page:
            logger.warning(
                f"Showing only first {self.config.max_clusters_per_page} clusters "
                f"(total: {len(cluster_ids)})"
            )
            cluster_ids = cluster_ids[:self.config.max_clusters_per_page]

        n_clusters = len(cluster_ids)
        n_cols = min(self.config.gallery_cols, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        # Create figure
        fig_width = n_cols * 3
        fig_height = n_rows * 3.5  # Extra space for captions

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            squeeze=False
        )

        # Flatten axes for easy iteration
        axes_flat = axes.flatten()

        for idx, cluster_id in enumerate(cluster_ids):
            ax = axes_flat[idx]
            info = results.cluster_info[cluster_id]

            # Load representative frame
            if info.representative_path:
                frame = self._load_representative_frame(
                    info.representative_path,
                    frame_base_dir
                )
            else:
                frame = None

            if frame is None:
                frame = self._create_placeholder_image(
                    f"Cluster {cluster_id}",
                    self.config.thumbnail_size
                )
            else:
                frame = frame.resize(self.config.thumbnail_size, Image.Resampling.LANCZOS)

            # Display frame
            ax.imshow(np.array(frame))
            ax.axis("off")

            # Add border based on cluster type
            border_color = "black"
            border_width = 2

            if info.is_micro_cluster:
                border_color = self.config.edge_case_color
                border_width = 4
            elif info.is_low_density:
                border_color = self.config.void_cluster_color
                border_width = 4

            # Add border rectangle
            rect = Rectangle(
                (0, 0),
                self.config.thumbnail_size[0] - 1,
                self.config.thumbnail_size[1] - 1,
                linewidth=border_width,
                edgecolor=border_color,
                facecolor="none"
            )
            ax.add_patch(rect)

            # Build caption
            caption_lines = [f"Cluster {cluster_id}"]
            caption_lines.append(f"Size: {info.size}")

            if info.is_micro_cluster:
                caption_lines.append("⚠️ Micro-cluster")
            if info.is_low_density:
                caption_lines.append("🔍 Low density")

            # Add VLM caption if available
            if self.config.show_captions and cluster_id in results.captions:
                vlm_caption = results.captions[cluster_id]
                # Truncate long captions
                if len(vlm_caption) > 50:
                    vlm_caption = vlm_caption[:47] + "..."
                caption_lines.append(f"📝 {vlm_caption}")

            caption = "\n".join(caption_lines)

            ax.set_title(
                caption,
                fontsize=8,
                pad=5,
                wrap=True
            )

        # Hide unused axes
        for idx in range(n_clusters, len(axes_flat)):
            axes_flat[idx].axis("off")

        # Add legend
        legend_text = (
            f"Total: {len(results.cluster_info)} clusters | "
            f"Edge cases: {len(results.edge_case_indices)} | "
            f"Void clusters: {len(results.void_cluster_ids)}"
        )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        fig.text(0.5, -0.02, legend_text, ha="center", fontsize=10)

        plt.tight_layout()

        # Save if requested
        if save_path is None and self.config.save_png:
            save_path = Path(self.config.output_dir) / "cluster_gallery.png"

        if save_path:
            fig.savefig(str(save_path), dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved gallery to {save_path}")

        return fig

    def create_html_gallery(
        self,
        results: "PipelineResults",
        frame_base_dir: Union[str, Path],
        title: str = "Interactive Cluster Gallery",
        save_path: Optional[str] = None
    ) -> str:
        """
        Create an interactive HTML gallery with embedded images.

        Args:
            results: Pipeline results.
            frame_base_dir: Base directory containing frame directories.
            title: Gallery title.
            save_path: Path to save HTML file.

        Returns:
            HTML string.
        """
        import base64
        from io import BytesIO
        from PIL import Image

        frame_base_dir = Path(frame_base_dir)

        # Sort clusters by size
        cluster_ids = sorted(
            results.cluster_info.keys(),
            key=lambda x: results.cluster_info[x].size,
            reverse=True
        )

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            "<style>",
            """
            body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
            h1 { text-align: center; color: #333; }
            .gallery { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }
            .card {
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                overflow: hidden;
                width: 250px;
                transition: transform 0.2s;
            }
            .card:hover { transform: translateY(-5px); box-shadow: 0 4px 16px rgba(0,0,0,0.15); }
            .card img { width: 100%; height: 200px; object-fit: cover; }
            .card-body { padding: 15px; }
            .card-title { font-weight: bold; margin-bottom: 5px; }
            .card-stat { font-size: 12px; color: #666; margin: 2px 0; }
            .card-caption { font-size: 11px; color: #888; margin-top: 8px; font-style: italic; }
            .badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 10px;
                margin: 2px;
            }
            .badge-edge { background: #FF6B6B; color: white; }
            .badge-void { background: #4ECDC4; color: white; }
            .summary {
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                background: white;
                border-radius: 10px;
            }
            """,
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
            "<div class='summary'>",
            f"<strong>Total Clusters:</strong> {len(results.cluster_info)} | ",
            f"<strong>Edge Cases:</strong> {len(results.edge_case_indices)} | ",
            f"<strong>Void Clusters:</strong> {len(results.void_cluster_ids)} | ",
            f"<strong>Total Samples:</strong> {len(results.cluster_labels)}",
            "</div>",
            "<div class='gallery'>"
        ]

        for cluster_id in cluster_ids:
            info = results.cluster_info[cluster_id]

            # Load and encode frame
            if info.representative_path:
                frame = self._load_representative_frame(
                    info.representative_path,
                    frame_base_dir
                )
            else:
                frame = None

            if frame is None:
                frame = self._create_placeholder_image(f"Cluster {cluster_id}")

            frame = frame.resize(self.config.thumbnail_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            frame.save(buffer, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Build card HTML
            badges = ""
            if info.is_micro_cluster:
                badges += "<span class='badge badge-edge'>Micro-cluster</span>"
            if info.is_low_density:
                badges += "<span class='badge badge-void'>Low Density</span>"

            caption = ""
            if cluster_id in results.captions:
                caption_text = results.captions[cluster_id]
                if len(caption_text) > 100:
                    caption_text = caption_text[:97] + "..."
                caption = f"<div class='card-caption'>{caption_text}</div>"

            card_html = f"""
            <div class='card'>
                <img src='data:image/jpeg;base64,{img_base64}' alt='Cluster {cluster_id}'>
                <div class='card-body'>
                    <div class='card-title'>Cluster {cluster_id}</div>
                    <div class='card-stat'>Size: {info.size} samples</div>
                    <div class='card-stat'>Density: {info.density:.4f}</div>
                    {badges}
                    {caption}
                </div>
            </div>
            """
            html_parts.append(card_html)

        html_parts.extend([
            "</div>",
            "</body>",
            "</html>"
        ])

        html_content = "\n".join(html_parts)

        # Save if requested
        if save_path is None and self.config.save_html:
            save_path = Path(self.config.output_dir) / "cluster_gallery.html"

        if save_path:
            Path(save_path).write_text(html_content)
            logger.info(f"Saved HTML gallery to {save_path}")

        return html_content

    def create_comparison_grid(
        self,
        results: "PipelineResults",
        frame_base_dir: Union[str, Path],
        cluster_ids: list[int],
        n_samples_per_cluster: int = 5,
        save_path: Optional[str] = None
    ) -> "matplotlib.figure.Figure":
        """
        Create a comparison grid showing multiple samples from selected clusters.

        Args:
            results: Pipeline results.
            frame_base_dir: Base directory containing frame directories.
            cluster_ids: List of cluster IDs to compare.
            n_samples_per_cluster: Number of samples to show per cluster.
            save_path: Path to save the image.

        Returns:
            Matplotlib Figure object.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        frame_base_dir = Path(frame_base_dir)

        n_clusters = len(cluster_ids)
        n_rows = n_clusters
        n_cols = n_samples_per_cluster

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 2.5, n_rows * 2.5),
            squeeze=False
        )

        # Get video paths for each cluster
        labels = results.cluster_labels

        for row_idx, cluster_id in enumerate(cluster_ids):
            # Get indices belonging to this cluster
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Sample indices
            if len(cluster_indices) > n_samples_per_cluster:
                sample_indices = np.random.choice(
                    cluster_indices,
                    n_samples_per_cluster,
                    replace=False
                )
            else:
                sample_indices = cluster_indices

            info = results.cluster_info[cluster_id]

            # Add row label
            axes[row_idx, 0].set_ylabel(
                f"Cluster {cluster_id}\n(n={info.size})",
                fontsize=10,
                rotation=0,
                ha="right",
                va="center"
            )

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]

                if col_idx < len(sample_indices):
                    idx = sample_indices[col_idx]

                    # Try to get video path
                    if hasattr(results, "video_paths") and results.video_paths:
                        video_path = results.video_paths[idx]
                    else:
                        video_path = f"video_{idx}.mp4"

                    frame = self._load_representative_frame(video_path, frame_base_dir)

                    if frame is None:
                        frame = self._create_placeholder_image("No Frame")

                    frame = frame.resize((150, 150), Image.Resampling.LANCZOS)
                    ax.imshow(np.array(frame))

                    # Highlight if edge case
                    if idx in results.edge_case_indices:
                        for spine in ax.spines.values():
                            spine.set_edgecolor(self.config.edge_case_color)
                            spine.set_linewidth(3)

                ax.axis("off")

        plt.suptitle("Cluster Comparison Grid", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Saved comparison grid to {save_path}")

        return fig
