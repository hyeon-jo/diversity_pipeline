from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from .base import CaptioningInterface
from ..embedder import frame_loader

logger = logging.getLogger(__name__)

class InternVLCaptioningInterface(CaptioningInterface):
    """
    InternVL3.5-8B based captioning interface with embedding caching support.

    This class provides hybrid captioning:
    1. If cached embeddings exist, use them directly for efficient inference
    2. If not, reload frames and generate captions from scratch

    The cached embeddings allow for faster captioning of representative videos
    without re-processing the visual features.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        embedding_cache_dir: Union[str, Path],
        frame_base_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        input_size: int = 448,
        num_frames: int = 16,
        torch_dtype: str = "bfloat16"
    ):
        """
        Initialize InternVL captioning interface.

        Args:
            model: Pre-loaded InternVL model (shared with embedder).
            tokenizer: Pre-loaded tokenizer.
            embedding_cache_dir: Directory containing cached .npy embeddings.
            frame_base_dir: Base directory for frame directories (for fallback).
            device: Device to use for inference.
            input_size: Input image size for InternVL.
            num_frames: Number of frames to sample for video captioning.
            torch_dtype: Torch dtype string ("bfloat16", "float16", "float32").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.frame_base_dir = Path(frame_base_dir) if frame_base_dir else None
        self.input_size = input_size
        self.num_frames = num_frames
        self.torch_dtype_str = torch_dtype

        # Set device
        if device is None:
            import torch
            self.device = next(model.parameters()).device
        else:
            import torch
            self.device = torch.device(device)

        # Initialize transform
        self._init_transform()

        logger.info(f"InternVLCaptioningInterface initialized")
        logger.info(f"  Embedding cache: {self.embedding_cache_dir}")
        logger.info(f"  Frame base dir: {self.frame_base_dir}")

    def _get_torch_dtype(self):
        """Get torch dtype from config string."""
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype_str, torch.bfloat16)

    def _init_transform(self) -> None:
        """Initialize the image transform for InternVL preprocessing."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        self._transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
            T.Resize((self.input_size, self.input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _get_embedding_path(self, video_path: str) -> Path:
        """
        Get the path to cached embedding file for a video.

        Args:
            video_path: Video path or video name.

        Returns:
            Path to the .npy embedding file.
        """
        video_name = Path(video_path).stem
        # Handle both .mp4 names and plain names
        if video_name.endswith(".mp4"):
            video_name = video_name[:-4]
        return self.embedding_cache_dir / f"{video_name}.npy"

    def _find_frame_directory(self, video_path: str) -> Optional[Path]:
        """
        Find the frame directory for a video.

        Searches for directories matching the video name pattern in frame_base_dir.

        Args:
            video_path: Video path or video name.

        Returns:
            Path to frame directory if found, None otherwise.
        """
        if self.frame_base_dir is None:
            return None

        video_name = Path(video_path).stem
        if video_name.endswith(".mp4"):
            video_name = video_name[:-4]

        # Search for matching frame directory
        # Pattern: N[숫자7자리]-[YYMMDDhhmmss]
        import re
        pattern = re.compile(rf'^{re.escape(video_name)}$')

        # Search in frame_base_dir recursively
        for dir_path in self.frame_base_dir.rglob("*"):
            if dir_path.is_dir() and pattern.match(dir_path.name):
                # Look for CMR_GT_Frame subdirectory
                frame_dir = dir_path / "RAW_DB"
                if frame_dir.exists():
                    cmr_dirs = list(frame_dir.glob("*_CMR*/CMR_GT_Frame"))
                    if cmr_dirs:
                        return cmr_dirs[0]

        return None

    def _caption_from_frames(
        self,
        frame_dir: Path,
        prompt: str,
        generation_config: Optional[dict] = None
    ) -> str:
        """
        Generate caption by loading frames and using InternVL chat.

        Args:
            frame_dir: Path to frame directory.
            prompt: Prompt for captioning.
            generation_config: Optional generation configuration.

        Returns:
            Generated caption string.
        """
        import torch

        # Load frames using frame_loader
        # Re-use the transform we initialized
        pixel_values, num_patches_list = frame_loader.load_frames_for_internvl(
            frame_dir, 
            self._transform,
            self.num_frames
        )

        # Move to device and dtype
        torch_dtype = self._get_torch_dtype()
        pixel_values = pixel_values.to(dtype=torch_dtype, device=self.device)

        # Default generation config
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 512,
                "do_sample": False,
            }

        # Build conversation with video placeholder
        # InternVL expects <image> tokens for each frame
        video_prefix = "".join(["<image>\n"] * len(num_patches_list))
        full_prompt = video_prefix + prompt

        # Use model.chat() for generation
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )

        return response

    def _caption_from_cached_embedding(
        self,
        embedding_path: Path,
        prompt: str,
        generation_config: Optional[dict] = None
    ) -> str:
        """
        Generate caption using cached vision embeddings.
        """
        import torch

        # Load cached embedding
        cached_embedding = np.load(embedding_path)

        # Convert to tensor
        torch_dtype = self._get_torch_dtype()
        vit_embeds = torch.from_numpy(cached_embedding).to(
            dtype=torch_dtype, device=self.device
        )

        # Reshape if needed: (hidden_dim,) -> (1, 1, hidden_dim)
        if vit_embeds.dim() == 1:
            vit_embeds = vit_embeds.unsqueeze(0).unsqueeze(0)
        elif vit_embeds.dim() == 2:
            vit_embeds = vit_embeds.unsqueeze(0)

        # Default generation config
        if generation_config is None:
            generation_config = {
                "max_new_tokens": 512,
                "do_sample": False,
            }

        # Try to use generate with pre-computed embeddings
        try:
            # Build input with placeholder
            video_prefix = "<image>\n"
            full_prompt = video_prefix + prompt

            # Tokenize prompt
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

            # Check if model supports direct embedding injection
            with torch.no_grad():
                # Try direct generation with cached embeddings
                if hasattr(self.model, 'generate_with_embeds'):
                    # Custom method if available
                    response = self.model.generate_with_embeds(
                        inputs.input_ids,
                        vit_embeds=vit_embeds,
                        **generation_config
                    )
                else:
                    raise NotImplementedError(
                        "Direct embedding injection not supported, "
                        "falling back to frame-based captioning"
                    )

            return self.tokenizer.decode(response[0], skip_special_tokens=True)

        except (NotImplementedError, AttributeError, TypeError) as e:
            logger.warning(f"Cached embedding captioning not supported: {e}")
            logger.warning("Falling back to frame-based captioning")
            return None  # Signal to use fallback

    def caption_video(
        self,
        video_path: str,
        cluster_id: int,
        prompt: str = "Describe this driving scenario in detail, including weather conditions, road type, traffic situation, and any notable events or hazards."
    ) -> str:
        """
        Generate caption for a video using InternVL.
        """
        # Try cached embedding first
        embedding_path = self._get_embedding_path(video_path)

        if embedding_path.exists():
            logger.info(f"Found cached embedding for {video_path}")
            caption = self._caption_from_cached_embedding(embedding_path, prompt)

            if caption is not None:
                return f"[Cluster {cluster_id}] {caption}"

        # Fallback: load frames and generate caption
        frame_dir = self._find_frame_directory(video_path)

        if frame_dir is not None and frame_dir.exists():
            logger.info(f"Loading frames from {frame_dir} for captioning")
            try:
                caption = self._caption_from_frames(frame_dir, prompt)
                return f"[Cluster {cluster_id}] {caption}"
            except Exception as e:
                logger.error(f"Frame-based captioning failed: {e}")

        # Ultimate fallback: return placeholder
        logger.warning(
            f"Could not generate caption for {video_path}: "
            f"no cached embedding or frame directory found"
        )
        return f"[Cluster {cluster_id}] Unable to generate caption - no cached embedding or frames available"

    def generate_cluster_captions(
        self,
        representative_videos: dict[int, str],
        prompt: str = "Describe this driving scenario in detail, including weather conditions, road type, traffic situation, and any notable events or hazards.",
        show_progress: bool = True
    ) -> dict[int, str]:
        """
        Generate captions for all representative videos.
        """
        captions = {}

        items = list(representative_videos.items())

        if show_progress:
            try:
                from tqdm import tqdm
                items = tqdm(items, desc="Generating captions")
            except ImportError:
                pass

        for cluster_id, video_path in items:
            captions[cluster_id] = self.caption_video(video_path, cluster_id, prompt)

        return captions