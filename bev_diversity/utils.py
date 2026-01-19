"""Utility functions for BEV diversity pipeline."""

from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray


def load_embeddings(embedding_path: Union[str, Path]) -> NDArray[np.float32]:
    """
    Load pre-computed embeddings from a numpy file.

    Supports:
    - .npy files: numpy array saved with np.save()
    - .npz files: compressed numpy archive saved with np.savez()

    Args:
        embedding_path: Path to .npy or .npz file

    Returns:
        Embedding matrix [N, D] as float32 array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid or embeddings have wrong shape

    Example:
        >>> embeddings = load_embeddings("embeddings.npy")
        >>> print(embeddings.shape)
        (1234, 1024)
    """
    embedding_path = Path(embedding_path)

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    print(f"Loading embeddings from {embedding_path}...")

    # Load based on file extension
    if embedding_path.suffix == ".npy":
        embeddings = np.load(embedding_path)

    elif embedding_path.suffix == ".npz":
        # Load npz archive
        data = np.load(embedding_path)

        # Try common keys
        if "embeddings" in data:
            embeddings = data["embeddings"]
        elif "arr_0" in data:
            embeddings = data["arr_0"]
        else:
            # Use first array in archive
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"NPZ file is empty: {embedding_path}")
            embeddings = data[keys[0]]
            print(f"  Using array '{keys[0]}' from NPZ file")

    else:
        raise ValueError(f"Unsupported file format: {embedding_path.suffix}")

    # Validate shape
    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D array [N, D], got shape {embeddings.shape}"
        )

    # Convert to float32
    embeddings = embeddings.astype(np.float32)

    print(f"Loaded {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")

    return embeddings


def save_embeddings(
    embeddings: NDArray[np.float32],
    output_path: Union[str, Path],
    compressed: bool = False,
) -> None:
    """
    Save embeddings to a numpy file.

    Args:
        embeddings: Embedding matrix [N, D]
        output_path: Path to save file (.npy or .npz)
        compressed: Whether to use compression (.npz format)

    Example:
        >>> embeddings = np.random.randn(100, 512).astype(np.float32)
        >>> save_embeddings(embeddings, "embeddings.npy")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if compressed or output_path.suffix == ".npz":
        np.savez_compressed(output_path, embeddings=embeddings)
        print(f"Embeddings saved (compressed) to: {output_path}")
    else:
        np.save(output_path, embeddings)
        print(f"Embeddings saved to: {output_path}")
