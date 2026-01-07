"""
Reference dataset handlers for BDD100K and nuImages.

Provides utilities to:
- Load and process reference dataset images
- Extract embeddings using the same model as the target dataset
- Cache embeddings for efficient reuse
- Sample subsets for faster evaluation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from video_curation_pipeline import VideoEmbedder

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a reference dataset."""
    name: str
    description: str
    n_samples: int
    embedding_dim: int
    source_path: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ReferenceDatasetLoader:
    """
    Handles loading and embedding extraction for reference datasets.

    Supports:
    - BDD100K: Berkeley Deep Drive dataset
    - nuImages: nuScenes image dataset
    - Custom: User-provided embeddings or image directories

    Example:
        loader = ReferenceDatasetLoader(embedder, cache_dir="./ref_cache")

        # Load pre-computed embeddings
        bdd_embeddings = loader.load_cached("bdd100k")

        # Or extract from images
        bdd_embeddings = loader.extract_embeddings(
            "bdd100k",
            image_dir="/path/to/bdd/images"
        )
    """

    # Dataset metadata
    DATASET_INFO = {
        "bdd100k": {
            "name": "BDD100K",
            "description": "Berkeley Deep Drive - 100K driving video frames",
            "url": "https://www.bdd100k.com/",
            "typical_size": 100000,
            "categories": ["weather", "scene", "timeofday"],
        },
        "nuimages": {
            "name": "nuImages",
            "description": "nuScenes image dataset - diverse driving scenes",
            "url": "https://www.nuscenes.org/nuimages",
            "typical_size": 93000,
            "categories": ["location", "weather", "lighting"],
        }
    }

    def __init__(
        self,
        embedder: Optional["VideoEmbedder"] = None,
        cache_dir: Union[str, Path] = "./reference_embeddings",
        device: str = "cuda"
    ):
        """
        Initialize the reference dataset loader.

        Args:
            embedder: VideoEmbedder instance for extracting embeddings.
            cache_dir: Directory to cache extracted embeddings.
            device: Device for embedding extraction.
        """
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self._loaded_datasets: dict[str, NDArray[np.float32]] = {}

    def get_cache_path(self, dataset_name: str) -> Path:
        """Get the cache file path for a dataset."""
        return self.cache_dir / f"{dataset_name}_embeddings.npy"

    def get_metadata_path(self, dataset_name: str) -> Path:
        """Get the metadata file path for a dataset."""
        return self.cache_dir / f"{dataset_name}_metadata.json"

    def load_cached(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None
    ) -> Optional[NDArray[np.float32]]:
        """
        Load cached embeddings for a reference dataset.

        Args:
            dataset_name: Name of the dataset ("bdd100k", "nuimages", or custom).
            max_samples: Maximum number of samples to load (random subset).

        Returns:
            Embeddings array or None if cache doesn't exist.
        """
        cache_path = self.get_cache_path(dataset_name)

        if not cache_path.exists():
            logger.warning(f"No cached embeddings found at {cache_path}")
            return None

        logger.info(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)

        # Subsample if requested
        if max_samples is not None and len(embeddings) > max_samples:
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            logger.info(f"Subsampled to {max_samples} embeddings")

        self._loaded_datasets[dataset_name] = embeddings
        logger.info(f"Loaded {len(embeddings)} embeddings for {dataset_name}")

        return embeddings

    def save_cached(
        self,
        dataset_name: str,
        embeddings: NDArray[np.float32],
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Save embeddings to cache.

        Args:
            dataset_name: Name of the dataset.
            embeddings: Embeddings to cache.
            metadata: Optional metadata to save alongside.
        """
        cache_path = self.get_cache_path(dataset_name)
        np.save(cache_path, embeddings)
        logger.info(f"Saved {len(embeddings)} embeddings to {cache_path}")

        if metadata:
            metadata_path = self.get_metadata_path(dataset_name)
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

    def extract_embeddings_bdd100k(
        self,
        image_dir: Union[str, Path],
        split: str = "train",
        max_samples: Optional[int] = 10000,
        save_cache: bool = True
    ) -> NDArray[np.float32]:
        """
        Extract embeddings from BDD100K images.

        BDD100K directory structure:
        bdd100k/
        ├── images/
        │   ├── 10k/
        │   │   ├── train/
        │   │   ├── val/
        │   │   └── test/
        │   └── 100k/
        │       ├── train/
        │       ├── val/
        │       └── test/
        └── labels/
            └── ...

        Args:
            image_dir: Path to BDD100K images directory.
            split: Dataset split ("train", "val", "test").
            max_samples: Maximum samples to process.
            save_cache: Whether to cache the extracted embeddings.

        Returns:
            Extracted embeddings array.
        """
        if self.embedder is None:
            raise ValueError("Embedder required for extraction. Initialize with embedder parameter.")

        image_dir = Path(image_dir)

        # Try different directory structures
        possible_paths = [
            image_dir / "images" / "100k" / split,
            image_dir / "images" / "10k" / split,
            image_dir / split,
            image_dir,
        ]

        images_path = None
        for path in possible_paths:
            if path.exists():
                images_path = path
                break

        if images_path is None:
            raise FileNotFoundError(f"Could not find BDD100K images in {image_dir}")

        logger.info(f"Loading BDD100K images from {images_path}")

        # Get image files
        image_files = list(images_path.glob("*.jpg"))
        if not image_files:
            image_files = list(images_path.glob("*.png"))

        if not image_files:
            raise FileNotFoundError(f"No images found in {images_path}")

        logger.info(f"Found {len(image_files)} images")

        # Subsample if needed
        if max_samples and len(image_files) > max_samples:
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]
            logger.info(f"Subsampled to {max_samples} images")

        # Extract embeddings
        embeddings = self._extract_from_images(image_files)

        # Save cache
        if save_cache:
            metadata = {
                "dataset": "bdd100k",
                "split": split,
                "n_samples": len(embeddings),
                "source_path": str(images_path),
            }
            self.save_cached("bdd100k", embeddings, metadata)

        return embeddings

    def extract_embeddings_nuimages(
        self,
        data_dir: Union[str, Path],
        split: str = "v1.0-train",
        max_samples: Optional[int] = 10000,
        save_cache: bool = True
    ) -> NDArray[np.float32]:
        """
        Extract embeddings from nuImages dataset.

        nuImages directory structure:
        nuimages/
        ├── samples/
        │   ├── CAM_FRONT/
        │   ├── CAM_FRONT_LEFT/
        │   ├── CAM_FRONT_RIGHT/
        │   ├── CAM_BACK/
        │   ├── CAM_BACK_LEFT/
        │   └── CAM_BACK_RIGHT/
        └── v1.0-train/
            └── ...json

        Args:
            data_dir: Path to nuImages data directory.
            split: Dataset split.
            max_samples: Maximum samples to process.
            save_cache: Whether to cache the extracted embeddings.

        Returns:
            Extracted embeddings array.
        """
        if self.embedder is None:
            raise ValueError("Embedder required for extraction.")

        data_dir = Path(data_dir)

        # Use front camera images (most relevant for driving)
        samples_dir = data_dir / "samples" / "CAM_FRONT"
        if not samples_dir.exists():
            samples_dir = data_dir / "samples"
        if not samples_dir.exists():
            samples_dir = data_dir

        logger.info(f"Loading nuImages from {samples_dir}")

        # Get image files
        image_files = list(samples_dir.rglob("*.jpg"))
        if not image_files:
            image_files = list(samples_dir.rglob("*.png"))

        if not image_files:
            raise FileNotFoundError(f"No images found in {samples_dir}")

        logger.info(f"Found {len(image_files)} images")

        # Subsample if needed
        if max_samples and len(image_files) > max_samples:
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]
            logger.info(f"Subsampled to {max_samples} images")

        # Extract embeddings
        embeddings = self._extract_from_images(image_files)

        # Save cache
        if save_cache:
            metadata = {
                "dataset": "nuimages",
                "split": split,
                "n_samples": len(embeddings),
                "source_path": str(samples_dir),
            }
            self.save_cached("nuimages", embeddings, metadata)

        return embeddings

    def _extract_from_images(
        self,
        image_files: list[Path],
        batch_size: int = 32
    ) -> NDArray[np.float32]:
        """
        Extract embeddings from a list of image files.

        Args:
            image_files: List of image file paths.
            batch_size: Batch size for processing.

        Returns:
            Embeddings array.
        """
        import torch
        from PIL import Image
        from tqdm import tqdm

        if not self.embedder._is_loaded:
            self.embedder.load_model()

        embeddings = []

        for i in tqdm(range(0, len(image_files), batch_size), desc="Extracting embeddings"):
            batch_files = image_files[i:i + batch_size]
            batch_embeddings = []

            for img_path in batch_files:
                try:
                    img = Image.open(img_path).convert("RGB")

                    # Apply transform
                    pixel_values = self.embedder._transform(img).unsqueeze(0)

                    # Extract features
                    with torch.no_grad():
                        vit_embeds = self.embedder._extract_vision_features(pixel_values)
                        embedding = vit_embeds.mean(dim=(0, 1))

                        if self.embedder.config.normalize_embeddings:
                            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

                        batch_embeddings.append(embedding.cpu().numpy())

                except Exception as e:
                    logger.warning(f"Failed to process {img_path}: {e}")
                    continue

            if batch_embeddings:
                embeddings.extend(batch_embeddings)

        return np.stack(embeddings).astype(np.float32)

    def load_or_extract(
        self,
        dataset_name: str,
        data_dir: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = 10000,
        force_extract: bool = False
    ) -> NDArray[np.float32]:
        """
        Load cached embeddings or extract if not available.

        Args:
            dataset_name: Name of dataset ("bdd100k", "nuimages", or custom).
            data_dir: Path to dataset images (required if not cached).
            max_samples: Maximum samples to use.
            force_extract: Force re-extraction even if cache exists.

        Returns:
            Embeddings array.
        """
        # Try cache first
        if not force_extract:
            embeddings = self.load_cached(dataset_name, max_samples)
            if embeddings is not None:
                return embeddings

        # Extract from images
        if data_dir is None:
            raise ValueError(f"No cached embeddings for {dataset_name}. Provide data_dir for extraction.")

        if dataset_name == "bdd100k":
            return self.extract_embeddings_bdd100k(data_dir, max_samples=max_samples)
        elif dataset_name == "nuimages":
            return self.extract_embeddings_nuimages(data_dir, max_samples=max_samples)
        else:
            # Custom dataset - assume flat directory of images
            return self.extract_embeddings_custom(dataset_name, data_dir, max_samples=max_samples)

    def extract_embeddings_custom(
        self,
        dataset_name: str,
        image_dir: Union[str, Path],
        max_samples: Optional[int] = None,
        save_cache: bool = True
    ) -> NDArray[np.float32]:
        """
        Extract embeddings from a custom image directory.

        Args:
            dataset_name: Name for the custom dataset.
            image_dir: Directory containing images.
            max_samples: Maximum samples to process.
            save_cache: Whether to cache the embeddings.

        Returns:
            Extracted embeddings array.
        """
        if self.embedder is None:
            raise ValueError("Embedder required for extraction.")

        image_dir = Path(image_dir)
        image_files = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(image_dir.rglob(ext))

        if not image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        logger.info(f"Found {len(image_files)} images in custom dataset")

        if max_samples and len(image_files) > max_samples:
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples]

        embeddings = self._extract_from_images(image_files)

        if save_cache:
            metadata = {
                "dataset": dataset_name,
                "n_samples": len(embeddings),
                "source_path": str(image_dir),
            }
            self.save_cached(dataset_name, embeddings, metadata)

        return embeddings

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a loaded or known dataset."""
        if dataset_name in self._loaded_datasets:
            embeddings = self._loaded_datasets[dataset_name]
            return DatasetInfo(
                name=dataset_name,
                description=self.DATASET_INFO.get(dataset_name, {}).get("description", "Custom dataset"),
                n_samples=len(embeddings),
                embedding_dim=embeddings.shape[1]
            )

        # Try to load metadata
        metadata_path = self.get_metadata_path(dataset_name)
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return DatasetInfo(
                name=dataset_name,
                description=self.DATASET_INFO.get(dataset_name, {}).get("description", "Custom dataset"),
                n_samples=metadata.get("n_samples", 0),
                embedding_dim=0,  # Unknown without loading
                source_path=metadata.get("source_path"),
                metadata=metadata
            )

        return None

    def list_available_datasets(self) -> list[str]:
        """List all available cached datasets."""
        cached = [p.stem.replace("_embeddings", "") for p in self.cache_dir.glob("*_embeddings.npy")]
        return sorted(set(cached))
