"""Image embedding extraction with progress bar."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm import tqdm

from bev_diversity.config import EmbeddingConfig, ModelConfig
from bev_diversity.embedder.model import LocalInternVLLoader


class ImageEmbedder:
    """
    Extract embeddings from images using InternVL model.

    Features:
    - Recursive image discovery with progress bar
    - Batch processing for efficiency
    - Optional L2 normalization
    """

    def __init__(self, model_config: ModelConfig, embedding_config: EmbeddingConfig):
        """
        Initialize image embedder.

        Args:
            model_config: Configuration for model loading
            embedding_config: Configuration for embedding extraction
        """
        self.model_config = model_config
        self.embedding_config = embedding_config

        # Initialize model loader
        self.loader = LocalInternVLLoader(
            model_path=model_config.model_path,
            torch_dtype=model_config.torch_dtype,
            device=model_config.device,
        )

        # Load model
        self.model, self.tokenizer, self.transform = self.loader.load()

    def discover_images(
        self,
        root_dir: Path,
        extensions: List[str],
        show_progress: bool = True,
    ) -> List[Path]:
        """
        Recursively discover all images in directory.

        Args:
            root_dir: Root directory to search
            extensions: List of file extensions to include (e.g., [".jpg", ".png"])
            show_progress: Whether to show progress bar

        Returns:
            Sorted list of image paths
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        print(f"Discovering images in {root_dir}...")

        # Collect all image files
        images = []
        for ext in extensions:
            # Use rglob for recursive search
            pattern = f"**/*{ext}"
            images.extend(root_dir.rglob(pattern))

        images = sorted(images)
        print(f"Found {len(images)} images")

        return images

    def extract_single(self, image_path: Path) -> NDArray[np.float32]:
        """
        Extract embedding for a single image.

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector as numpy array [D]
        """
        # Preprocess image
        pixel_values = self.loader.preprocess_image(image_path)

        # Extract features
        features = self.loader.extract_vision_features(pixel_values)

        # Convert to numpy
        embedding = features.squeeze(0).numpy().astype(np.float32)

        # Normalize if requested
        if self.embedding_config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def extract_batch(
        self,
        image_paths: List[Path],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """
        Extract embeddings for multiple images with progress bar.

        Args:
            image_paths: List of image paths
            show_progress: Whether to show progress bar

        Returns:
            Embedding matrix [N, D] where N is number of images
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        embeddings = []
        batch_size = self.embedding_config.batch_size

        # Process in batches
        num_batches = (len(image_paths) + batch_size - 1) // batch_size

        with tqdm(
            total=len(image_paths),
            desc="Extracting embeddings",
            disable=not show_progress,
        ) as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]

                # Preprocess batch
                batch_pixels = []
                for path in batch_paths:
                    try:
                        pixel_values = self.loader.preprocess_image(path)
                        batch_pixels.append(pixel_values)
                    except Exception as e:
                        print(f"\nWarning: Failed to load {path}: {e}")
                        continue

                if not batch_pixels:
                    continue

                # Stack into batch tensor
                batch_tensor = torch.cat(batch_pixels, dim=0)

                # Extract features
                try:
                    features = self.loader.extract_vision_features(batch_tensor)
                    embeddings.append(features.numpy().astype(np.float32))
                except Exception as e:
                    print(f"\nWarning: Failed to extract features: {e}")
                    continue

                pbar.update(len(batch_paths))

        if not embeddings:
            raise RuntimeError("No embeddings extracted")

        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)

        # Normalize if requested
        if self.embedding_config.normalize_embeddings:
            norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)  # Avoid division by zero
            all_embeddings = all_embeddings / norms

        return all_embeddings

    def extract_from_directory(
        self,
        root_dir: Path,
        extensions: List[str],
        show_progress: bool = True,
    ) -> NDArray[np.float32]:
        """
        Discover and extract embeddings from all images in directory.

        Args:
            root_dir: Root directory containing images
            extensions: List of file extensions to include
            show_progress: Whether to show progress bar

        Returns:
            Embedding matrix [N, D]
        """
        # Discover images
        image_paths = self.discover_images(root_dir, extensions, show_progress)

        if not image_paths:
            raise RuntimeError(f"No images found in {root_dir}")

        # Extract embeddings
        embeddings = self.extract_batch(image_paths, show_progress)

        return embeddings
