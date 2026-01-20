"""Embedding extraction subpackage."""

from bev_diversity.embedder.model import LocalInternVLLoader
from bev_diversity.embedder.extractor import ImageEmbedder
from bev_diversity.embedder.video_embedder import VideoEmbedder, load_embeddings_from_dir
from bev_diversity.embedder.frame_loader import (
    get_frame_indices,
    discover_video_folders,
    discover_frames_in_folder,
    group_frames_single_folder,
    load_frames_tensor,
)

__all__ = [
    "LocalInternVLLoader",
    "ImageEmbedder",
    "VideoEmbedder",
    "load_embeddings_from_dir",
    "get_frame_indices",
    "discover_video_folders",
    "discover_frames_in_folder",
    "group_frames_single_folder",
    "load_frames_tensor",
]
