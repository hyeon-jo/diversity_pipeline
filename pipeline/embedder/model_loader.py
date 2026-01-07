from __future__ import annotations
import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from ..config import VideoEmbedderConfig

logger = logging.getLogger(__name__)

class InternVLModelLoader:
    """
    Handles loading of InternVL model and tokenizer, and feature extraction.
    """
    def __init__(self, config: VideoEmbedderConfig):
        self.config = config
        self.device = None
        self.model = None
        self.tokenizer = None
        self.transform = None
        self.is_loaded = False

    def _ensure_torch(self) -> None:
        """Ensure torch is imported and device is set."""
        if self.device is None:
            import torch
            self.device = torch.device(self.config.device)

    def get_torch_dtype(self):
        """Get torch dtype from config string."""
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.torch_dtype, torch.bfloat16)

    def load_model(self) -> None:
        """Load the InternVL3.5-8B model and tokenizer from HuggingFace."""
        if self.is_loaded:
            logger.info("Model already loaded, skipping...")
            return

        self._ensure_torch()
        import torch

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading InternVL model: {self.config.model_name}")
            logger.info(f"Device: {self.device}, Dtype: {self.config.torch_dtype}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )

            # Load model with specified dtype
            torch_dtype = self.get_torch_dtype()

            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=self.config.use_flash_attn,
                trust_remote_code=self.config.trust_remote_code
            ).to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")

            # Initialize transform for frame preprocessing
            self._init_transform()

        except ImportError as e:
            raise ImportError(
                "transformers library required. Install with: "
                "pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _init_transform(self) -> None:
        """Initialize the image transform for InternVL preprocessing."""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
            T.Resize((self.config.input_size, self.config.input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def extract_vision_features(
        self,
        pixel_values: Any
    ) -> Any:
        """
        Extract vision features using InternVL's vision encoder only.

        Args:
            pixel_values: Preprocessed pixel values tensor of shape (N, C, H, W).

        Returns:
            Vision embeddings tensor.
        """
        import torch

        # Move to device and set dtype
        torch_dtype = self.get_torch_dtype()
        pixel_values = pixel_values.to(dtype=torch_dtype, device=self.device)

        # Use InternVL's extract_feature method (vision encoder only)
        # This extracts features without running the full LLM
        with torch.no_grad():
            vit_embeds = self.model.extract_feature(pixel_values)

        return vit_embeds