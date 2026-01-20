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
    min_frames: int = 1,
    show_progress: bool = True
) -> List[Path]:
    """
    Recursively discover all subdirectories that contain frame images (video folders).

    Args:
        root_dir: Root directory to search.
        min_frames: Minimum number of frames required to be considered a video folder.
        show_progress: Whether to show progress logs during discovery.

    Returns:
        Sorted list of video folder paths.

    Example:
        >>> discover_video_folders(Path("/data/videos"))
        [Path('/data/videos/video1'), Path('/data/videos/sub/video2'), ...]
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise ValueError(f"Root directory does not exist: {root_dir}")

    video_folders = []
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    if show_progress:
        print(f"[Discovery] Scanning directories under: {root_dir}")

    # Recursively find all directories
    all_dirs = list(root_dir.rglob("*"))
    all_dirs = [d for d in all_dirs if d.is_dir()]
    # Also include the root directory itself
    all_dirs = [root_dir] + all_dirs
    total_dirs = len(all_dirs)

    if show_progress:
        print(f"[Discovery] Found {total_dirs} directories to scan")

    # Check each directory for image files
    for idx, subdir in enumerate(sorted(all_dirs)):
        if show_progress and (idx + 1) % 100 == 0:
            print(f"[Discovery] Scanning directory {idx + 1}/{total_dirs}: {subdir.name}")

        # Count image files in this directory (not recursive, just this folder)
        try:
            image_count = sum(
                1 for f in subdir.iterdir()
                if f.is_file() and f.suffix in image_extensions
            )
        except PermissionError:
            if show_progress:
                print(f"[Discovery] Warning: Permission denied for {subdir}")
            continue

        if image_count >= min_frames:
            video_folders.append(subdir)

    video_folders = sorted(video_folders)

    if show_progress:
        print(f"[Discovery] Complete! Found {len(video_folders)} video folders with images")

    return video_folders


def discover_frames_in_folder(
    folder: Path,
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
    show_progress: bool = True
) -> List[Path]:
    """
    Discover and sort frame files in a folder.

    Args:
        folder: Folder containing frame images.
        extensions: List of valid extensions. Defaults to [".jpg", ".jpeg", ".png"].
        recursive: Whether to search recursively in subdirectories.
        show_progress: Whether to show progress logs during discovery.

    Returns:
        Sorted list of frame file paths.
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    if show_progress:
        search_type = "recursively" if recursive else "in folder"
        print(f"[Frames] Discovering frames {search_type}: {folder}")

    # Collect all frame files
    frames = []
    glob_method = folder.rglob if recursive else folder.glob
    pattern_prefix = "" if recursive else ""

    for ext in extensions:
        if recursive:
            frames.extend(folder.rglob(f"*{ext}"))
            frames.extend(folder.rglob(f"*{ext.upper()}"))
        else:
            frames.extend(folder.glob(f"*{ext}"))
            frames.extend(folder.glob(f"*{ext.upper()}"))

    # Sort by filename
    frames = sorted(set(frames))  # Remove duplicates from case variations

    if show_progress:
        print(f"[Frames] Found {len(frames)} frame files")

    return frames


def group_frames_single_folder(
    root_dir: Path,
    extensions: Optional[List[str]] = None,
    frames_per_video: Optional[int] = None,
    recursive: bool = False,
    show_progress: bool = True
) -> Dict[str, List[Path]]:
    """
    Group frames in a single folder (or recursively) into videos.

    Two strategies:
    1. If frames_per_video is specified: Split sorted frames into groups of N
    2. Otherwise: Group by common filename prefix

    Args:
        root_dir: Folder containing all frame images.
        extensions: List of valid extensions.
        frames_per_video: Number of frames per video (for fixed grouping).
        recursive: Whether to search recursively in subdirectories.
        show_progress: Whether to show progress logs during discovery.

    Returns:
        Dictionary mapping video names to lists of frame paths.

    Example:
        >>> group_frames_single_folder(Path("/data/frames"), frames_per_video=100)
        {'video_0': [frame1.jpg, ...], 'video_1': [frame101.jpg, ...]}
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    root_dir = Path(root_dir)

    if show_progress:
        print(f"[Grouping] Starting frame grouping from: {root_dir}")

    # Get all frames sorted
    all_frames = discover_frames_in_folder(root_dir, extensions, recursive=recursive, show_progress=show_progress)

    if not all_frames:
        return {}

    if frames_per_video is not None:
        # Strategy 1: Fixed grouping by count
        if show_progress:
            print(f"[Grouping] Grouping {len(all_frames)} frames into groups of {frames_per_video}")
        groups = {}
        for i in range(0, len(all_frames), frames_per_video):
            video_name = f"video_{i // frames_per_video:04d}"
            groups[video_name] = all_frames[i:i + frames_per_video]
        if show_progress:
            print(f"[Grouping] Complete! Created {len(groups)} video groups")
        return groups
    else:
        # Strategy 2: Group by common prefix
        # Try to find common prefix pattern in filenames
        if show_progress:
            print(f"[Grouping] Grouping {len(all_frames)} frames by filename prefix")
        groups = {}
        import re
        for frame_path in all_frames:
            # Extract prefix (everything before last underscore or number sequence)
            name = frame_path.stem
            # Simple heuristic: use everything up to last numeric sequence
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

        if show_progress:
            print(f"[Grouping] Complete! Created {len(groups)} video groups")

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
