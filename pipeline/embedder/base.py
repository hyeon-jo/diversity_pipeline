"""Main VideoEmbedder class for extracting embeddings using InternVL3.5-8B."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ..config import VideoEmbedderConfig
from . import frame_loader
from . import model_loader

logger = logging.getLogger(__name__)


class VideoEmbedder:
    """
    Extract dense vector representations from video clips using InternVL3.5-8B.

    This class handles:
    - Loading the InternVL3.5-8B model and tokenizer
    - Frame sampling with dynamic resolution preprocessing
    - Vision encoder feature extraction (without LLM forward)
    - L2 normalization of embeddings
    - Caching embeddings for later captioning
    """

    def __init__(self, config: Optional[VideoEmbedderConfig] = None):
        """
        Initialize the VideoEmbedder.

        Args:
            config: Configuration for the embedder. Uses defaults if None.
        """
        self.config = config or VideoEmbedderConfig()
        self.device = None  # Lazily initialized
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._transform = None
        self._model_loader = None

    def _ensure_torch(self) -> None:
        """Ensure torch is imported and device is set."""
        if self.device is None:
            import torch
            self.device = torch.device(self.config.device)

    def load_model(self) -> None:
        """Load the InternVL3.5-8B model and tokenizer from HuggingFace."""
        if self._is_loaded:
            logger.info("Model already loaded, skipping...")
            return

        self._ensure_torch()

        # Create model loader
        self._model_loader = model_loader.InternVLModelLoader(
            model_name=self.config.model_name,
            input_size=self.config.input_size,
            torch_dtype=self.config.torch_dtype,
            use_flash_attn=self.config.use_flash_attn,
            trust_remote_code=self.config.trust_remote_code,
            device=str(self.device)
        )

        # Load model, tokenizer, and transform
        self.model, self.tokenizer, self._transform = self._model_loader.load_model()
        self._is_loaded = True

    def extract_embedding(
        self,
        video_path: Union[str, Path]
    ) -> NDArray[np.float32]:
        """
        Extract embedding for a single video file.

        Args:
            video_path: Path to the video file.

        Returns:
            L2-normalized embedding vector.

        Raises:
            RuntimeError: If video cannot be processed.
        """
        import torch

        if not self._is_loaded:
            self.load_model()

        try:
            # Sample frames from video
            frames = frame_loader.sample_frames_decord(
                video_path,
                num_frames=self.config.num_frames,
                strategy=self.config.frame_sample_strategy
            )

            # Convert numpy frames to tensor with transforms
            from PIL import Image
            pixel_values_list = []
            for i in range(frames.shape[0]):
                img = Image.fromarray(frames[i])
                pv = self._transform(img).unsqueeze(0)
                pixel_values_list.append(pv)

            pixel_values = torch.cat(pixel_values_list, dim=0)

            # Extract vision features
            vit_embeds = self._model_loader.extract_vision_features(
                self.model, pixel_values
            )

            # Mean pooling over all patches and frames
            # vit_embeds shape: (num_frames, num_patches, hidden_dim)
            embedding = vit_embeds.mean(dim=(0, 1))  # (hidden_dim,)

            # Normalize if configured
            if self.config.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

            return embedding.cpu().numpy().astype(np.float32)

        except Exception as e:
            raise RuntimeError(
                f"Failed to process video {video_path}: {e}"
            ) from e

    def extract_embeddings_batch(
        self,
        video_paths: list[Union[str, Path]],
        show_progress: bool = True
    ) -> tuple[NDArray[np.float32], list[int]]:
        """
        Extract embeddings for multiple videos with batch processing.

        Args:
            video_paths: List of paths to video files.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (embeddings array, list of failed indices).
        """
        if not self._is_loaded:
            self.load_model()

        embeddings = []
        failed_indices = []

        # Optional progress bar
        iterator = video_paths
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(video_paths, desc="Extracting embeddings")
            except ImportError:
                pass

        for idx, video_path in enumerate(iterator):
            try:
                embedding = self.extract_embedding(video_path)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to process {video_path}: {e}")
                failed_indices.append(idx)

        if not embeddings:
            raise RuntimeError("No videos could be processed successfully")

        return np.stack(embeddings), failed_indices

    def extract_embedding_from_frame_dir(
        self,
        frame_dir: Union[str, Path],
        save_embedding: bool = True
    ) -> tuple[NDArray[np.float32], str]:
        """
        Extract embedding from a directory of frame JPG files using InternVL vision encoder.

        Args:
            frame_dir: Path to directory containing frame JPG files.
            save_embedding: Whether to save the embedding to output directory.

        Returns:
            Tuple of (embedding vector, video name).
        """
        import torch

        if not self._is_loaded:
            self.load_model()

        frame_dir = Path(frame_dir)
        video_name = frame_loader.extract_video_name_from_frame_dir(frame_dir)

        try:
            # Load frames with InternVL preprocessing
            pixel_values, num_patches_list = frame_loader.load_frames_for_internvl(
                frame_dir,
                transform=self._transform,
                num_frames=self.config.num_frames,
                strategy=self.config.frame_sample_strategy
            )

            # Extract vision features using InternVL's vision encoder
            vit_embeds = self._model_loader.extract_vision_features(
                self.model, pixel_values
            )

            # Mean pooling over all patches and frames
            # vit_embeds shape: (num_frames, num_patches, hidden_dim)
            embedding = vit_embeds.mean(dim=(0, 1))  # (hidden_dim,)

            # Normalize if configured
            if self.config.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

            embedding_f32 = embedding.to(torch.float32)
            embedding_np = embedding_f32.cpu().numpy().astype(np.float32)

            # Save embedding if requested
            if save_embedding:
                self._save_embedding(embedding_np, video_name)

            return embedding_np, video_name

        except Exception as e:
            raise RuntimeError(
                f"Failed to process frame directory {frame_dir}: {e}"
            ) from e

    def _save_embedding(
        self,
        embedding: NDArray[np.float32],
        video_name: str
    ) -> Path:
        """
        Save embedding to output directory with video name.

        Args:
            embedding: Embedding vector to save.
            video_name: Original video filename (e.g., "N1234567-231215120000.mp4")

        Returns:
            Path to saved embedding file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Replace .mp4 with .npy
        embedding_filename = video_name.replace(".mp4", ".npy")
        output_path = output_dir / embedding_filename

        np.save(output_path, embedding)
        logger.debug(f"Saved embedding to {output_path}")

        return output_path

    def extract_embeddings_from_frame_dirs(
        self,
        frame_dirs: list[Union[str, Path]],
        save_embeddings: bool = True,
        show_progress: bool = True
    ) -> tuple[NDArray[np.float32], list[str], list[int]]:
        """
        Extract embeddings from multiple frame directories.

        Args:
            frame_dirs: List of paths to frame directories.
            save_embeddings: Whether to save embeddings to output directory.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (embeddings array, video names, failed indices).
        """
        if not self._is_loaded:
            self.load_model()

        embeddings = []
        video_names = []
        failed_indices = []

        # Ensure output directory exists
        if save_embeddings:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Optional progress bar
        iterator = enumerate(frame_dirs)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Extracting embeddings from frames")
            except ImportError:
                iterator = enumerate(frame_dirs)

        for idx, frame_dir in iterator:
            try:
                embedding, video_name = self.extract_embedding_from_frame_dir(
                    frame_dir, save_embedding=save_embeddings
                )
                embeddings.append(embedding)
                video_names.append(video_name)

            except Exception as e:
                logger.warning(f"Failed to process {frame_dir}: {e}")
                failed_indices.append(idx)

        if not embeddings:
            raise RuntimeError("No frame directories could be processed successfully")

        return np.stack(embeddings), video_names, failed_indices

    # Expose frame_loader functions as static methods for backward compatibility
    @staticmethod
    def find_frame_directories(
        base_dir: Union[str, Path],
        pattern: str = "**/CMR_GT_Frame"
    ) -> list[Path]:
        """
        Find all frame directories matching the expected structure (legacy glob method).

        Expected structure:
        ./trainlake/[yy].[mm]w/N[숫자7자리]-[YYMMDDhhmmss]/RAW_DB/*_CMR*/CMR_GT_Frame/

        Args:
            base_dir: Base directory to search from.
            pattern: Glob pattern to match frame directories.

        Returns:
            List of paths to frame directories.

        Note:
            This uses Path.glob() which can be slow. Consider using
            find_frame_directories_optimized() instead.
        """
        return frame_loader.find_frame_directories(base_dir, pattern)

    @staticmethod
    def find_frame_directories_optimized(
        base_dir: Union[str, Path],
        target_name: str = "CMR_GT_Frame",
        progress_callback: Optional[Any] = None
    ) -> list[Path]:
        """
        Find frame directories using optimized 3-stage traversal.

        This method is 10-12x faster than Path.glob("**/CMR_GT_Frame").

        Args:
            base_dir: Base directory to search from.
            target_name: Target directory name to find.
            progress_callback: Optional callback(current_dir: str, count: int).

        Returns:
            List of paths to frame directories, sorted by path.
        """
        return frame_loader.find_frame_directories_optimized(
            base_dir, target_name, progress_callback
        )

    @staticmethod
    def find_frame_directories_cached(
        base_dir: Union[str, Path],
        cache_file: Optional[Path] = None,
        force_rescan: bool = False,
        progress_callback: Optional[Any] = None
    ) -> list[Path]:
        """
        Find frame directories with caching support.

        Args:
            base_dir: Base directory to search from.
            cache_file: Path to cache file (default: base_dir/.frame_dirs_cache.txt).
            force_rescan: If True, ignore cache and rescan directories.
            progress_callback: Optional callback for progress updates.

        Returns:
            List of paths to frame directories.
        """
        return frame_loader.find_frame_directories_cached(
            base_dir, cache_file, force_rescan, progress_callback
        )

    @staticmethod
    def extract_video_name_from_frame_dir(frame_dir: Union[str, Path]) -> str:
        """
        Extract original video name from frame directory path.

        Args:
            frame_dir: Path to the frame directory.

        Returns:
            Original video filename (e.g., "N1234567-231215120000.mp4")
        """
        return frame_loader.extract_video_name_from_frame_dir(frame_dir)
