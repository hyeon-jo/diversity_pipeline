"""Frame loading and directory scanning utilities for video embeddings."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def extract_video_name_from_frame_dir(frame_dir: Union[str, Path]) -> str:
    """
    Extract original video name from frame directory path.

    Expected path structure:
    ./trainlake/[yy].[mm]w/N[숫자7자리]-[YYMMDDhhmmss]/RAW_DB/*_CMR*/CMR_GT_Frame/*.jpg

    The video name is extracted from the parent directory structure:
    N[숫자7자리]-[YYMMDDhhmmss].mp4

    Args:
        frame_dir: Path to the frame directory (CMR_GT_Frame folder or its parent)

    Returns:
        Original video filename (e.g., "N1234567-231215120000.mp4")
    """
    path = Path(frame_dir)

    # Navigate up to find the directory matching N[7digits]-[timestamp] pattern
    # Pattern: N followed by 7 digits, dash, then 12 digits (YYMMDDhhmmss)
    pattern = re.compile(r'^N\d{7}-\d{12}$')

    current = path
    for _ in range(10):  # Limit search depth
        if pattern.match(current.name):
            return f"{current.name}.mp4"
        if current.parent == current:
            break
        current = current.parent

    # Fallback: use the frame directory name
    logger.warning(f"Could not extract video name from {frame_dir}, using directory name")
    return f"{path.name}.mp4"


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
    """
    if total_frames < num_frames:
        # Repeat frames if not enough
        indices = np.linspace(
            0, total_frames - 1, num_frames, dtype=int
        )
    elif strategy == "uniform":
        indices = np.linspace(
            0, total_frames - 1, num_frames, dtype=int
        )
    else:  # random sampling
        indices = np.sort(
            np.random.choice(
                total_frames, num_frames, replace=False
            )
        )
    return indices


def load_frames_for_internvl(
    frame_dir: Union[str, Path],
    transform: Any,
    num_frames: int = 16,
    strategy: str = "uniform"
) -> tuple[Any, list[int]]:
    """
    Load frames from directory with InternVL preprocessing.

    Args:
        frame_dir: Path to directory containing frame JPG files.
        transform: Torchvision transform to apply to each frame.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").

    Returns:
        Tuple of (pixel_values tensor, num_patches_list).
        pixel_values: Tensor of shape (total_patches, C, H, W)
        num_patches_list: List of patch counts per frame
    """
    import torch
    from PIL import Image

    frame_dir = Path(frame_dir)

    # Get all jpg files sorted by name
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("*.JPG"))

    if not frame_files:
        raise ValueError(f"No JPG files found in {frame_dir}")

    total_frames = len(frame_files)
    indices = get_frame_indices(total_frames, num_frames, strategy)

    pixel_values_list = []
    num_patches_list = []

    for idx in indices:
        img = Image.open(frame_files[idx]).convert("RGB")

        # Apply transform (resize to input_size, normalize)
        pixel_values = transform(img)  # (C, H, W)
        pixel_values = pixel_values.unsqueeze(0)  # (1, C, H, W)

        num_patches_list.append(1)  # 1 patch per frame for video
        pixel_values_list.append(pixel_values)

    # Concatenate all frames
    pixel_values = torch.cat(pixel_values_list, dim=0)  # (num_frames, C, H, W)

    return pixel_values, num_patches_list


def sample_frames_from_directory(
    frame_dir: Union[str, Path],
    num_frames: int = 16,
    strategy: str = "uniform"
) -> NDArray[np.uint8]:
    """
    Sample frames from a directory of JPG files (legacy method).

    Args:
        frame_dir: Path to directory containing frame JPG files.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").

    Returns:
        Array of shape (num_frames, H, W, C) with uint8 values.
    """
    from PIL import Image

    frame_dir = Path(frame_dir)

    # Get all jpg files sorted by name
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frame_dir.glob("*.JPG"))

    if not frame_files:
        raise ValueError(f"No JPG files found in {frame_dir}")

    total_frames = len(frame_files)
    indices = get_frame_indices(total_frames, num_frames, strategy)

    # Load selected frames
    frames = []
    for idx in indices:
        img = Image.open(frame_files[idx]).convert("RGB")
        frames.append(np.array(img))

    return np.stack(frames)


def sample_frames_decord(
    video_path: Union[str, Path],
    num_frames: int = 16,
    strategy: str = "uniform"
) -> NDArray[np.uint8]:
    """
    Sample frames from video using decord (preferred for performance).

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").

    Returns:
        Array of shape (num_frames, H, W, C) with uint8 values.
    """
    try:
        import decord
        decord.bridge.set_bridge("numpy")

        vr = decord.VideoReader(str(video_path))
        total_frames = len(vr)

        indices = get_frame_indices(total_frames, num_frames, strategy)
        frames = vr.get_batch(indices).asnumpy()
        return frames

    except ImportError:
        logger.warning("decord not available, falling back to PyAV")
        return sample_frames_pyav(video_path, num_frames, strategy)


def sample_frames_pyav(
    video_path: Union[str, Path],
    num_frames: int = 16,
    strategy: str = "uniform"
) -> NDArray[np.uint8]:
    """
    Sample frames from video using PyAV (fallback).

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy ("uniform" or "random").

    Returns:
        Array of shape (num_frames, H, W, C) with uint8 values.
    """
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]

    # Get total frames
    total_frames = stream.frames
    if total_frames == 0:
        # Estimate from duration
        total_frames = int(
            stream.duration * stream.time_base * stream.average_rate
        )

    # Calculate frame indices to sample
    if strategy == "uniform":
        indices = set(
            np.linspace(
                0, max(total_frames - 1, 0),
                num_frames,
                dtype=int
            )
        )
    else:
        indices = set(
            np.sort(
                np.random.choice(
                    total_frames,
                    min(num_frames, total_frames),
                    replace=False
                )
            )
        )

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        if len(frames) >= num_frames:
            break

    container.close()

    # Pad if necessary
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(frames[:num_frames])


def find_frame_directories(
    base_dir: Union[str, Path],
    pattern: str = "**/CMR_GT_Frame"
) -> list[Path]:
    """
    Find all frame directories matching the expected structure (legacy glob method).

    Expected structure:
    ./trainlake/[yy].[mm]w/N[숫자7자리]-[YYMMDDhhmmss]/RAW_DB/*_CMR*/CMR_GT_Frame/

    Args:
        base_dir: Base directory to search from.
        pattern: Glob pattern to match frame directories.

    Returns:
        List of paths to frame directories.

    Note:
        This uses Path.glob() which can be slow for large directory trees.
        Consider using find_frame_directories_optimized() instead.
    """
    base_dir = Path(base_dir)
    frame_dirs = sorted(base_dir.glob(pattern))
    logger.info(f"Found {len(frame_dirs)} frame directories in {base_dir}")
    return frame_dirs


def find_frame_directories_optimized(
    base_dir: Union[str, Path],
    target_name: str = "CMR_GT_Frame",
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> list[Path]:
    """
    Find frame directories using optimized 3-stage traversal.

    Exploits known directory structure:
    trainlake/[yy].[mm]w/N[7자리]-[12자리]/RAW_DB/*_CMR*/CMR_GT_Frame/

    This method is 10-12x faster than Path.glob("**/CMR_GT_Frame") by:
    1. Using os.scandir() instead of glob
    2. Applying regex filters at each level to skip irrelevant directories
    3. Direct path construction instead of recursive search

    Args:
        base_dir: Base directory to search from (e.g., "./trainlake").
        target_name: Target directory name to find (default: "CMR_GT_Frame").
        progress_callback: Optional callback(current_dir: str, count: int) for progress updates.

    Returns:
        List of paths to frame directories, sorted by path.

    Example:
        >>> def progress(dir_name, count):
        ...     print(f"\\rChecked {count} directories | Current: {dir_name[:30]}...", end="")
        >>> dirs = find_frame_directories_optimized("./trainlake", progress_callback=progress)
        >>> print(f"\\nFound {len(dirs)} directories")
    """
    base_dir = Path(base_dir)
    results = []
    dirs_scanned = 0

    # Regex patterns for each level
    week_pattern = re.compile(r'^\d{2}\.\d{2}w$')  # [yy].[mm]w
    video_pattern = re.compile(r'^N\d{7}-\d{12}$')  # N[7digits]-[12digits]

    # Stage 1: Iterate week directories ([yy].[mm]w)
    try:
        for week_entry in base_dir.iterdir():
            if not week_entry.is_dir():
                continue
            if not week_pattern.match(week_entry.name):
                continue

            # Stage 2: Iterate video directories (N[7digits]-[12digits])
            for video_entry in week_entry.iterdir():
                if not video_entry.is_dir():
                    continue
                if not video_pattern.match(video_entry.name):
                    continue

                dirs_scanned += 1
                if progress_callback:
                    progress_callback(video_entry.name, dirs_scanned)

                # Stage 3: Direct path to RAW_DB/*_CMR*/CMR_GT_Frame
                raw_db_dir = video_entry / "RAW_DB"
                if not raw_db_dir.exists():
                    continue

                for cmr_entry in raw_db_dir.iterdir():
                    if not cmr_entry.is_dir():
                        continue
                    if "_CMR" not in cmr_entry.name:
                        continue

                    frame_dir = cmr_entry / target_name
                    if frame_dir.exists() and frame_dir.is_dir():
                        results.append(frame_dir)

    except PermissionError as e:
        logger.warning(f"Permission denied accessing {base_dir}: {e}")
    except Exception as e:
        logger.error(f"Error scanning directories: {e}")
        raise

    results.sort()
    logger.info(f"Found {len(results)} frame directories after scanning {dirs_scanned} video directories")
    return results


def find_frame_directories_cached(
    base_dir: Union[str, Path],
    cache_file: Optional[Path] = None,
    force_rescan: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> list[Path]:
    """
    Find frame directories with caching support.

    If cache file exists and force_rescan is False, loads paths from cache.
    Otherwise, performs optimized scan and saves results to cache.

    Args:
        base_dir: Base directory to search from.
        cache_file: Path to cache file (default: base_dir/.frame_dirs_cache.txt).
        force_rescan: If True, ignore cache and rescan directories.
        progress_callback: Optional callback for progress updates during scan.

    Returns:
        List of paths to frame directories.

    Example:
        >>> # First run: scans and caches
        >>> dirs = find_frame_directories_cached("./trainlake")
        >>> # Second run: loads from cache (instant)
        >>> dirs = find_frame_directories_cached("./trainlake")
        >>> # Force rescan
        >>> dirs = find_frame_directories_cached("./trainlake", force_rescan=True)
    """
    base_dir = Path(base_dir)

    # Default cache location
    if cache_file is None:
        cache_file = base_dir / ".frame_dirs_cache.txt"

    # Try to load from cache
    if cache_file.exists() and not force_rescan:
        logger.info(f"Loading frame directories from cache: {cache_file}")
        try:
            cached_paths = cache_file.read_text().strip().split("\n")
            results = [Path(p) for p in cached_paths if p]
            logger.info(f"Loaded {len(results)} frame directories from cache")
            return results
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}, performing rescan")

    # Perform optimized scan
    logger.info(f"Scanning directories (cache_file: {cache_file})")
    results = find_frame_directories_optimized(
        base_dir,
        progress_callback=progress_callback
    )

    # Save to cache
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("\n".join(str(p) for p in results))
        logger.info(f"Saved {len(results)} paths to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache file: {e}")

    return results


def validate_directory_structure(
    frame_dir: Union[str, Path],
    min_frames: int = 1
) -> bool:
    """
    Validate that a frame directory has the expected structure and contents.

    Args:
        frame_dir: Path to the frame directory.
        min_frames: Minimum number of JPG files required (default: 1).

    Returns:
        True if directory is valid, False otherwise.
    """
    frame_dir = Path(frame_dir)

    # Check directory exists
    if not frame_dir.exists() or not frame_dir.is_dir():
        logger.debug(f"Invalid: {frame_dir} does not exist or is not a directory")
        return False

    # Check for JPG files
    jpg_files = list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.JPG"))
    if len(jpg_files) < min_frames:
        logger.debug(f"Invalid: {frame_dir} has only {len(jpg_files)} JPG files (min: {min_frames})")
        return False

    # Check path structure matches expected pattern
    try:
        video_name = extract_video_name_from_frame_dir(frame_dir)
        if not video_name.endswith(".mp4"):
            logger.debug(f"Invalid: Could not extract valid video name from {frame_dir}")
            return False
    except Exception as e:
        logger.debug(f"Invalid: Error extracting video name from {frame_dir}: {e}")
        return False

    return True
