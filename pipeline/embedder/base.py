from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from ..config import VideoEmbedderConfig
from .model_loader import InternVLModelLoader
from . import frame_loader

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

class VideoEmbedder:
    """
    Extract dense vector representations from video clips using InternVL3.5-8B.
    
    Delegates model loading to InternVLModelLoader and frame loading to frame_loader module.
    """

    def __init__(self, config: Optional[VideoEmbedderConfig] = None):
        """
        Initialize the VideoEmbedder.

        Args:
            config: Configuration for the embedder. Uses defaults if None.
        """
        self.config = config or VideoEmbedderConfig()
        self.loader = InternVLModelLoader(self.config)

    @property
    def model(self):
        return self.loader.model

    @property
    def tokenizer(self):
        return self.loader.tokenizer
        
    @property
    def device(self):
        return self.loader.device

    @property
    def _is_loaded(self):
        return self.loader.is_loaded

    def load_model(self) -> None:
        """Load the InternVL3.5-8B model and tokenizer."""
        self.loader.load_model()

    @staticmethod
    def extract_video_name_from_frame_dir(frame_dir: Union[str, Path]) -> str:
        return frame_loader.extract_video_name_from_frame_dir(frame_dir)

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
        """
        import torch

        if not self._is_loaded:
            self.load_model()

        try:
            # Sample frames from video
            frames = frame_loader.sample_frames_decord(
                video_path, 
                self.config.num_frames, 
                self.config.frame_sample_strategy
            )

            # Convert numpy frames to tensor with transforms
            from PIL import Image
            pixel_values_list = []
            for i in range(frames.shape[0]):
                img = Image.fromarray(frames[i])
                pv = self.loader.transform(img).unsqueeze(0)
                pixel_values_list.append(pv)

            pixel_values = torch.cat(pixel_values_list, dim=0)

            # Extract vision features
            vit_embeds = self.loader.extract_vision_features(pixel_values)

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

        batch_frames = []
        batch_indices = []
        
        # Use batch_size from config or default to 1
        batch_size = getattr(self.config, 'batch_size', 1)

        for idx, video_path in enumerate(iterator):
            try:
                frames = frame_loader.sample_frames_decord(
                    video_path,
                    self.config.num_frames,
                    self.config.frame_sample_strategy
                )
                batch_frames.append(frames)
                batch_indices.append(idx)

                # Process batch when full
                if len(batch_frames) >= batch_size:
                    batch_embeddings = self._process_batch(batch_frames)
                    embeddings.extend(batch_embeddings)
                    batch_frames = []
                    batch_indices = []

            except Exception as e:
                logger.warning(f"Failed to process {video_path}: {e}")
                failed_indices.append(idx)

        # Process remaining batch
        if batch_frames:
            batch_embeddings = self._process_batch(batch_frames)
            embeddings.extend(batch_embeddings)

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
        video_name = self.extract_video_name_from_frame_dir(frame_dir)

        try:
            # Load frames with InternVL preprocessing
            pixel_values, num_patches_list = frame_loader.load_frames_for_internvl(
                frame_dir, 
                self.loader.transform,
                self.config.num_frames,
                self.config.frame_sample_strategy
            )

            # Extract vision features using InternVL's vision encoder
            vit_embeds = self.loader.extract_vision_features(pixel_values)

            # Mean pooling over all patches and frames
            embedding = vit_embeds.mean(dim=(0, 1))  # (hidden_dim,)

            # Normalize if configured
            if self.config.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

            embedding_np = embedding.cpu().numpy().astype(np.float32)

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
        """Save embedding to output directory with video name."""
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

    @staticmethod
    def find_frame_directories(
        base_dir: Union[str, Path],
        pattern: str = "**/CMR_GT_Frame"
    ) -> list[Path]:
        return frame_loader.find_frame_directories(base_dir, pattern)
        
    @staticmethod
    def find_frame_directories_optimized(
        base_dir: Union[str, Path],
        target_name: str = "CMR_GT_Frame",
        progress_callback: Optional[Any] = None
    ) -> list[Path]:
        return frame_loader.find_frame_directories_optimized(
            base_dir, target_name, progress_callback
        )
        
    @staticmethod
    def find_frame_directories_cached(
        base_dir: Union[str, Path],
        cache_file: Optional[Union[str, Path]] = None,
        force_rescan: bool = False,
        progress_callback: Optional[Any] = None
    ) -> list[Path]:
        return frame_loader.find_frame_directories_cached(
            base_dir, cache_file, force_rescan, progress_callback
        )

    def _process_batch(
        self,
        batch_frames: list[NDArray[np.uint8]]
    ) -> list[NDArray[np.float32]]:
        """Process a batch of frame arrays through InternVL vision encoder."""
        import torch
        from PIL import Image

        embeddings = []

        for frames in batch_frames:
            # Convert numpy frames to tensor with transforms
            pixel_values_list = []
            for i in range(frames.shape[0]):
                img = Image.fromarray(frames[i])
                pv = self.loader.transform(img).unsqueeze(0)
                pixel_values_list.append(pv)

            pixel_values = torch.cat(pixel_values_list, dim=0)

            # Extract vision features
            vit_embeds = self.loader.extract_vision_features(pixel_values)

            # Mean pooling over all patches and frames
            embedding = vit_embeds.mean(dim=(0, 1))

            if self.config.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

            embeddings.append(embedding.cpu().numpy().astype(np.float32))

        return embeddings