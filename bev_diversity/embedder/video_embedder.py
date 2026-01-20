"""Video embedding extraction with frame pooling and memory optimization."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from bev_diversity.config import ModelConfig, VideoConfig, InputConfig, OutputConfig
from bev_diversity.embedder.model import LocalInternVLLoader
from bev_diversity.embedder.frame_loader import (
    discover_video_folders,
    discover_frames_in_folder,
    group_frames_single_folder,
    load_frames_tensor,
    get_frame_indices,
)


class VideoEmbedder:
    """
    Extract video embeddings using frame pooling.

    Features:
    - Samples N frames uniformly from each video
    - Extracts frame embeddings using InternVL vision encoder
    - Pools frame embeddings (mean/max) to get single video embedding
    - Memory-optimized: saves each embedding immediately to disk
    - Resume support: skips videos with existing embeddings
    """

    def __init__(
        self,
        model_config: ModelConfig,
        video_config: VideoConfig,
        input_config: Optional[InputConfig] = None,
        output_config: Optional[OutputConfig] = None,
    ):
        """
        Initialize video embedder.

        Args:
            model_config: Configuration for model loading
            video_config: Configuration for video processing
            input_config: Configuration for input structure (optional)
            output_config: Configuration for output/memory (optional)
        """
        self.model_config = model_config
        self.video_config = video_config
        self.input_config = input_config or InputConfig()
        self.output_config = output_config or OutputConfig()

        # Initialize model loader
        self.loader = LocalInternVLLoader(
            model_path=model_config.model_path,
            torch_dtype=model_config.torch_dtype,
            device=model_config.device,
        )

        # Load model
        self.model, self.tokenizer, self.transform = self.loader.load()

    def extract_video_embedding(
        self,
        frame_paths: List[Path],
        video_name: Optional[str] = None,
    ) -> NDArray[np.float32]:
        """
        Extract single video embedding from frame files using pooling.

        Args:
            frame_paths: List of all frame file paths for this video.
            video_name: Optional name for logging.

        Returns:
            Video embedding vector of shape (D,).
        """
        if not frame_paths:
            raise ValueError("frame_paths cannot be empty")

        num_frames = self.video_config.num_frames
        strategy = self.video_config.frame_sample_strategy

        # Get sample indices
        total_frames = len(frame_paths)
        indices = get_frame_indices(total_frames, num_frames, strategy)

        # Load and transform sampled frames
        pixel_values = load_frames_tensor(
            frame_paths,
            self.transform,
            num_frames,
            strategy,
        )

        # Extract vision features for all frames
        # pixel_values: (num_frames, C, H, W)
        # vit_embeds: depends on model, typically (num_frames, num_patches, hidden_dim)
        #             or (num_frames, hidden_dim) for pooled output
        with torch.no_grad():
            pixel_values = pixel_values.to(
                self.loader.device,
                dtype=self.loader.torch_dtype
            )

            if hasattr(self.model, 'extract_feature'):
                vit_embeds = self.model.extract_feature(pixel_values)
            elif hasattr(self.model, 'vision_model'):
                vit_embeds = self.model.vision_model(pixel_values).pooler_output
            else:
                outputs = self.model(pixel_values=pixel_values, return_dict=True)
                if hasattr(outputs, 'vision_outputs'):
                    vit_embeds = outputs.vision_outputs.pooler_output
                else:
                    raise RuntimeError("Cannot extract vision features from this model")

        # Pool across frames (and patches if present)
        # vit_embeds shape could be:
        # - (num_frames, hidden_dim) - already pooled per frame
        # - (num_frames, num_patches, hidden_dim) - need to pool patches too
        if vit_embeds.dim() == 3:
            # (num_frames, num_patches, hidden_dim) -> pool patches and frames
            if self.video_config.pooling_strategy == "mean":
                embedding = vit_embeds.mean(dim=(0, 1))  # (hidden_dim,)
            else:  # max
                embedding = vit_embeds.max(dim=1)[0].max(dim=0)[0]  # (hidden_dim,)
        else:
            # (num_frames, hidden_dim) -> pool frames only
            if self.video_config.pooling_strategy == "mean":
                embedding = vit_embeds.mean(dim=0)  # (hidden_dim,)
            else:  # max
                embedding = vit_embeds.max(dim=0)[0]  # (hidden_dim,)

        # Normalize if configured
        if self.video_config.normalize_embeddings:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

        return embedding.cpu().float().numpy()

    def extract_and_save_embeddings(
        self,
        video_folders: List[Path],
        output_dir: Path,
        extensions: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Tuple[List[Path], List[str]]:
        """
        Extract embeddings for all videos and save each immediately to disk.

        Memory-optimized: only one embedding in memory at a time.
        Resume-capable: skips videos with existing embedding files.

        Args:
            video_folders: List of video folder paths (each containing frames).
            output_dir: Directory to save embedding files.
            extensions: Valid image extensions.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (list of saved embedding paths, list of video names).
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]

        # Create embedding subdirectory
        embedding_dir = output_dir / self.output_config.embedding_subdir
        embedding_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        video_names = []
        skipped_count = 0

        iterator = video_folders
        if show_progress:
            iterator = tqdm(video_folders, desc="Extracting video embeddings")

        for video_folder in iterator:
            video_name = video_folder.name
            output_path = embedding_dir / f"{video_name}.npy"

            # Skip if already exists and skip_existing is enabled
            if self.output_config.skip_existing and output_path.exists():
                skipped_count += 1
                saved_paths.append(output_path)
                video_names.append(video_name)
                continue

            try:
                # Discover frames in this video folder (no recursive, no progress for inner loop)
                frame_paths = discover_frames_in_folder(video_folder, extensions, recursive=False, show_progress=False)

                if not frame_paths:
                    print(f"\nWarning: No frames found in {video_folder}, skipping")
                    continue

                # Extract embedding
                embedding = self.extract_video_embedding(frame_paths, video_name)

                # Save immediately to disk
                np.save(output_path, embedding)

                saved_paths.append(output_path)
                video_names.append(video_name)

                # Update progress bar description
                if show_progress and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(
                        frames=len(frame_paths),
                        saved=len(saved_paths),
                    )

            except Exception as e:
                print(f"\nError processing {video_folder}: {e}")
                continue

        if skipped_count > 0:
            print(f"Skipped {skipped_count} videos with existing embeddings")

        return saved_paths, video_names

    def extract_from_video_folders(
        self,
        root_dir: Path,
        output_dir: Path,
        extensions: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Tuple[List[Path], List[str]]:
        """
        Extract embeddings from video folder structure.

        Expected structure:
        root_dir/
        ├── video1/
        │   ├── frame001.jpg
        │   └── ...
        ├── video2/
        │   └── ...

        Args:
            root_dir: Root directory containing video folders.
            output_dir: Directory to save embedding files.
            extensions: Valid image extensions.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (list of embedding file paths, list of video names).
        """
        root_dir = Path(root_dir)

        video_folders = discover_video_folders(root_dir, show_progress=show_progress)

        if not video_folders:
            raise RuntimeError(f"No video folders found in {root_dir}")

        return self.extract_and_save_embeddings(
            video_folders,
            output_dir,
            extensions,
            show_progress,
        )

    def extract_from_single_folder(
        self,
        root_dir: Path,
        output_dir: Path,
        frames_per_video: Optional[int] = None,
        extensions: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Tuple[List[Path], List[str]]:
        """
        Extract embeddings from single folder with grouped frames.

        Args:
            root_dir: Folder containing all frame images.
            output_dir: Directory to save embedding files.
            frames_per_video: Number of frames per video for grouping.
            extensions: Valid image extensions.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (list of embedding file paths, list of video names).
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png"]

        root_dir = Path(root_dir)

        frame_groups = group_frames_single_folder(
            root_dir,
            extensions,
            frames_per_video or self.input_config.frames_per_video,
            recursive=self.input_config.recursive,
            show_progress=show_progress,
        )

        if not frame_groups:
            raise RuntimeError(f"No frame groups found in {root_dir}")

        # Create embedding subdirectory
        embedding_dir = output_dir / self.output_config.embedding_subdir
        embedding_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        video_names = []
        skipped_count = 0

        iterator = sorted(frame_groups.keys())
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting video embeddings")

        for video_name in iterator:
            frame_paths = frame_groups[video_name]
            output_path = embedding_dir / f"{video_name}.npy"

            # Skip if already exists
            if self.output_config.skip_existing and output_path.exists():
                skipped_count += 1
                saved_paths.append(output_path)
                video_names.append(video_name)
                continue

            try:
                # Extract embedding
                embedding = self.extract_video_embedding(frame_paths, video_name)

                # Save immediately
                np.save(output_path, embedding)

                saved_paths.append(output_path)
                video_names.append(video_name)

            except Exception as e:
                print(f"\nError processing {video_name}: {e}")
                continue

        if skipped_count > 0:
            print(f"Skipped {skipped_count} videos with existing embeddings")

        return saved_paths, video_names


def load_embeddings_from_dir(
    embedding_dir: Path,
    show_progress: bool = True,
) -> Tuple[NDArray[np.float32], List[str]]:
    """
    Load all embeddings from a directory of .npy files.

    Args:
        embedding_dir: Directory containing .npy embedding files.
        show_progress: Whether to show progress bar.

    Returns:
        Tuple of (embeddings array [N, D], list of video names).
    """
    embedding_dir = Path(embedding_dir)
    if not embedding_dir.exists():
        raise ValueError(f"Embedding directory does not exist: {embedding_dir}")

    # Find all .npy files
    embedding_files = sorted(embedding_dir.glob("*.npy"))
    if not embedding_files:
        raise ValueError(f"No .npy files found in {embedding_dir}")

    embeddings = []
    video_names = []

    iterator = embedding_files
    if show_progress:
        iterator = tqdm(embedding_files, desc="Loading embeddings")

    for emb_file in iterator:
        embedding = np.load(emb_file)
        embeddings.append(embedding)
        video_names.append(emb_file.stem)

    # Stack into single array
    embeddings_array = np.stack(embeddings)

    print(f"Loaded {len(embeddings)} embeddings (dimension: {embeddings_array.shape[1]})")

    return embeddings_array, video_names
