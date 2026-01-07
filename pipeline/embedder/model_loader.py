"""InternVL model loading and initialization utilities."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class InternVLModelLoader:
    """Handles loading and initialization of InternVL3.5-8B model."""

    # ImageNet normalization constants
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL3_5-8B",
        input_size: int = 448,
        torch_dtype: str = "bfloat16",
        use_flash_attn: bool = True,
        trust_remote_code: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize the model loader.

        Args:
            model_name: HuggingFace model identifier.
            input_size: Input image resolution (default: 448 for InternVL).
            torch_dtype: Torch dtype string ("bfloat16", "float16", or "float32").
            use_flash_attn: Whether to use flash attention.
            trust_remote_code: Whether to trust remote code (required for OpenGVLab).
            device: Device to load model on ("cuda" or "cpu").
        """
        self.model_name = model_name
        self.input_size = input_size
        self.torch_dtype = torch_dtype
        self.use_flash_attn = use_flash_attn
        self.trust_remote_code = trust_remote_code
        self.device = device

        self.model = None
        self.tokenizer = None
        self.transform = None

    def get_torch_dtype(self):
        """
        Get torch dtype from config string.

        Returns:
            Corresponding torch dtype object.
        """
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def init_transform(self) -> Any:
        """
        Initialize the image transform for InternVL preprocessing.

        Returns:
            Composed torchvision transform.
        """
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'convert') else img),
            T.Resize(
                (self.input_size, self.input_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        return transform

    def load_model(self) -> tuple[Any, Any, Any]:
        """
        Load the InternVL3.5-8B model and tokenizer from HuggingFace.

        Returns:
            Tuple of (model, tokenizer, transform).

        Raises:
            ImportError: If transformers library is not installed.
            RuntimeError: If model loading fails.
        """
        import torch

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading InternVL model: {self.model_name}")
            logger.info(f"Device: {self.device}, Dtype: {self.torch_dtype}")

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )

            # Load model with specified dtype
            torch_dtype = self.get_torch_dtype()

            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=self.use_flash_attn,
                trust_remote_code=self.trust_remote_code
            ).to(self.device)
            model.eval()

            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(
                f"Model parameters: "
                f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B"
            )

            # Initialize transform for frame preprocessing
            transform = self.init_transform()

            self.model = model
            self.tokenizer = tokenizer
            self.transform = transform

            return model, tokenizer, transform

        except ImportError as e:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def extract_vision_features(
        self,
        model: Any,
        pixel_values: Any,
        device: Optional[str] = None
    ) -> Any:
        """
        Extract vision features using InternVL's vision encoder only.

        Args:
            model: Loaded InternVL model.
            pixel_values: Preprocessed pixel values tensor of shape (N, C, H, W).
            device: Device to use (uses self.device if None).

        Returns:
            Vision embeddings tensor.
        """
        import torch

        if device is None:
            device = self.device

        # Move to device and set dtype
        torch_dtype = self.get_torch_dtype()
        pixel_values = pixel_values.to(dtype=torch_dtype, device=device)

        # Use InternVL's extract_feature method (vision encoder only)
        # This extracts features without running the full LLM
        with torch.no_grad():
            vit_embeds = model.extract_feature(pixel_values)

        return vit_embeds


def load_internvl_model(
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    input_size: int = 448,
    torch_dtype: str = "bfloat16",
    use_flash_attn: bool = True,
    trust_remote_code: bool = True,
    device: str = "cuda"
) -> tuple[Any, Any, Any]:
    """
    Convenience function to load InternVL model.

    Args:
        model_name: HuggingFace model identifier.
        input_size: Input image resolution.
        torch_dtype: Torch dtype string ("bfloat16", "float16", or "float32").
        use_flash_attn: Whether to use flash attention.
        trust_remote_code: Whether to trust remote code.
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer, transform).
    """
    loader = InternVLModelLoader(
        model_name=model_name,
        input_size=input_size,
        torch_dtype=torch_dtype,
        use_flash_attn=use_flash_attn,
        trust_remote_code=trust_remote_code,
        device=device
    )
    return loader.load_model()


def extract_vision_features(
    model: Any,
    pixel_values: Any,
    device: str = "cuda",
    torch_dtype: str = "bfloat16"
) -> Any:
    """
    Convenience function to extract vision features.

    Args:
        model: Loaded InternVL model.
        pixel_values: Preprocessed pixel values tensor of shape (N, C, H, W).
        device: Device to use.
        torch_dtype: Torch dtype string.

    Returns:
        Vision embeddings tensor.
    """
    loader = InternVLModelLoader(device=device, torch_dtype=torch_dtype)
    return loader.extract_vision_features(model, pixel_values, device)
