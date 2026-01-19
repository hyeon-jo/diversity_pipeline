"""InternVL model loading from local filesystem path."""

from pathlib import Path
from typing import Any, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class LocalInternVLLoader:
    """
    Load InternVL model from a LOCAL filesystem path.

    Unlike the HuggingFace approach, this loader requires the model
    to be already downloaded and stored locally.
    """

    def __init__(self, model_path: str, torch_dtype: str = "bfloat16", device: str = "cuda"):
        """
        Initialize model loader.

        Args:
            model_path: Local path to InternVL model directory
            torch_dtype: Data type for model ("bfloat16", "float16", "float32")
            device: Device to load model on ("cuda" or "cpu")
        """
        self.model_path = Path(model_path)
        self.torch_dtype = self._get_torch_dtype(torch_dtype)
        self.device = torch.device(device)

        self.model = None
        self.tokenizer = None
        self.transform = None

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_str not in dtype_map:
            raise ValueError(f"Invalid dtype: {dtype_str}")
        return dtype_map[dtype_str]

    def load(self) -> Tuple[Any, Any, Any]:
        """
        Load model, tokenizer, and image transform from local path.

        Returns:
            Tuple of (model, tokenizer, transform)

        Raises:
            ValueError: If model path doesn't exist or model loading fails
        """
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

        print(f"Loading InternVL model from {self.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            local_files_only=True,
        )

        # Load model
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device).eval()

        # Create image transform (ImageNet normalization)
        self.transform = T.Compose([
            T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"Model loaded successfully on {self.device}")

        return self.model, self.tokenizer, self.transform

    @torch.no_grad()
    def extract_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract vision features using InternVL's vision encoder.

        Args:
            pixel_values: Preprocessed image tensor [B, 3, H, W]

        Returns:
            Vision features tensor [B, D] where D is embedding dimension
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pixel_values = pixel_values.to(self.device, dtype=self.torch_dtype)

        # Extract features using vision encoder
        # InternVL3.5 has a method to extract vision features
        if hasattr(self.model, 'extract_feature'):
            features = self.model.extract_feature(pixel_values)
        elif hasattr(self.model, 'vision_model'):
            features = self.model.vision_model(pixel_values).pooler_output
        else:
            # Fallback: try forward pass and get vision outputs
            outputs = self.model(pixel_values=pixel_values, return_dict=True)
            if hasattr(outputs, 'vision_outputs'):
                features = outputs.vision_outputs.pooler_output
            else:
                raise RuntimeError("Cannot extract vision features from this model")

        return features.cpu().float()

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess a single image.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        if self.transform is None:
            raise RuntimeError("Transform not initialized. Call load() first.")

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.transform(image).unsqueeze(0)
        return pixel_values
