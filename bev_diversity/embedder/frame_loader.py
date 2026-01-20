"""Frame loading and sampling utilities for video embeddings."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def get_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: str = "uniform"
) -> NDArray[np.int64]:
    """
    Get frame indices to sample based on strategy.

    Args:
        total_frames: Total number of frames available.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").

    Returns:
        Array of frame indices to sample.

    Example:
        >>> get_frame_indices(100, 16, "uniform")
        array([ 0,  6, 13, 20, 26, 33, 40, 46, 53, 60, 66, 73, 80, 86, 93, 99])
    """
    if total_frames <= num_frames:
        # If not enough frames, use all available (with possible repetition)
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    elif strategy == "uniform":
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    elif strategy == "random":
        indices = np.sort(
            np.random.choice(total_frames, num_frames, replace=False)
        )
    else:
        raise ValueError(f"Invalid sampling strategy: {strategy}")

    return indices


def discover_video_folders(
    root_dir: Path,
    min_frames: int = 1
) -> List[Path]:
    """
    Discover subdirectories that contain frame images (video folders).

    Args:
        root_dir: Root directory to search.
        min_frames: Minimum number of frames required to be considered a video folder.

    Returns:
        Sorted list of video folder paths.

    Example:
        >>> discover_video_folders(Path("/data/videos"))
        [Path('/data/videos/video1'), Path('/data/videos/video2'), ...]
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    video_folders = []
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    # Check immediate subdirectories
    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Count image files in this directory
        image_count = sum(
            1 for f in subdir.iterdir()
            if f.is_file() and f.suffix in image_extensions
        )

        if image_count >= min_frames:
            video_folders.append(subdir)

    return video_folders


def discover_frames_in_folder(
    folder: Path,
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Discover and sort frame files in a folder.

    Args:
        folder: Folder containing frame images.
        extensions: List of valid extensions. Defaults to [".jpg", ".jpeg", ".png"].

    Returns:
        Sorted list of frame file paths.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    # Collect all frame files
    frames = []
    for ext in extensions:
        frames.extend(folder.glob(f"*{ext}"))
        frames.extend(folder.glob(f"*{ext.upper()}"))

    # Sort by filename
    frames = sorted(set(frames))  # Remove duplicates from case variations

    return frames


def group_frames_single_folder(
    root_dir: Path,
    extensions: Optional[List[str]] = None,
    frames_per_video: Optional[int] = None
) -> Dict[str, List[Path]]:
    """
    Group frames in a single folder into videos.

    Two strategies:
    1. If frames_per_video is specified: Split sorted frames into groups of N
    2. Otherwise: Group by common filename prefix

    Args:
        root_dir: Folder containing all frame images.
        extensions: List of valid extensions.
        frames_per_video: Number of frames per video (for fixed grouping).

    Returns:
        Dictionary mapping video names to lists of frame paths.

    Example:
        >>> group_frames_single_folder(Path("/data/frames"), frames_per_video=100)
        {'video_0': [frame1.jpg, ...], 'video_1': [frame101.jpg, ...]}
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    root_dir = Path(root_dir)

    # Get all frames sorted
    all_frames = discover_frames_in_folder(root_dir, extensions)

    if not all_frames:
        return {}

    if frames_per_video is not None:
        # Strategy 1: Fixed grouping by count
        groups = {}
        for i in range(0, len(all_frames), frames_per_video):
            video_name = f"video_{i // frames_per_video:04d}"
            groups[video_name] = all_frames[i:i + frames_per_video]
        return groups
    else:
        # Strategy 2: Group by common prefix
        # Try to find common prefix pattern in filenames
        groups = {}
        for frame_path in all_frames:
            # Extract prefix (everything before last underscore or number sequence)
            name = frame_path.stem
            # Simple heuristic: use everything up to last numeric sequence
            import re
            match = re.match(r'^(.+?)[\d_\-\.]+$', name)
            if match:
                prefix = match.group(1).rstrip('_-.')
            else:
                prefix = name

            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(frame_path)

        # Sort frames within each group
        for prefix in groups:
            groups[prefix] = sorted(groups[prefix])

        return groups


def load_frames_tensor(
    frame_paths: List[Path],
    transform: Any,
    num_frames: int,
    strategy: str = "uniform"
) -> Any:
    """
    Load frames and convert to tensor with sampling.

    Args:
        frame_paths: List of all frame file paths.
        transform: Torchvision transform to apply.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").

    Returns:
        Tensor of shape (num_frames, C, H, W).
    """
    import torch
    from PIL import Image

    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")

    # Get sample indices
    total_frames = len(frame_paths)
    indices = get_frame_indices(total_frames, num_frames, strategy)

    # Load and transform frames
    pixel_values_list = []
    for idx in indices:
        img = Image.open(frame_paths[idx]).convert("RGB")
        pixel_values = transform(img)  # (C, H, W)
        pixel_values = pixel_values.unsqueeze(0)  # (1, C, H, W)
        pixel_values_list.append(pixel_values)

    # Concatenate all frames
    pixel_values = torch.cat(pixel_values_list, dim=0)  # (num_frames, C, H, W)

    return pixel_values


def load_frames_numpy(
    frame_paths: List[Path],
    num_frames: int,
    strategy: str = "uniform",
    target_size: Optional[Tuple[int, int]] = None
) -> NDArray[np.uint8]:
    """
    Load frames as numpy array with sampling.

    Args:
        frame_paths: List of all frame file paths.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").
        target_size: Optional (width, height) to resize frames.

    Returns:
        Array of shape (num_frames, H, W, C) with uint8 values.
    """
    from PIL import Image

    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")

    # Get sample indices
    total_frames = len(frame_paths)
    indices = get_frame_indices(total_frames, num_frames, strategy)

    # Load frames
    frames = []
    for idx in indices:
        img = Image.open(frame_paths[idx]).convert("RGB")
        if target_size is not None:
            img = img.resize(target_size, Image.BILINEAR)
        frames.append(np.array(img))

    return np.stack(frames)
