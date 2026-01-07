from __future__ import annotations
import re
import logging
from pathlib import Path
from typing import Any, Optional, Union, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch
    from ..config import VideoEmbedderConfig

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
        strategy: "uniform" or "random".

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
    num_frames: int,
    strategy: str = "uniform"
) -> tuple[Any, list[int]]:
    """
    Load frames from directory with InternVL preprocessing.

    Args:
        frame_dir: Path to directory containing frame JPG files.
        transform: Torchvision transform to apply.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy.

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
    num_frames: int,
    strategy: str = "uniform"
) -> NDArray[np.uint8]:
    """
    Sample frames from a directory of JPG files (legacy method).

    Args:
        frame_dir: Path to directory containing frame JPG files.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy.

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
    num_frames: int,
    strategy: str = "uniform"
) -> NDArray[np.uint8]:
    """
    Sample frames from video using decord (preferred for performance).

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy.

    Returns:
        Array of shape (num_frames, H, W, C) with uint8 values.
    """
    try:
        import decord
        decord.bridge.set_bridge("numpy")

        vr = decord.VideoReader(str(video_path))
        total_frames = len(vr)

        if total_frames < num_frames:
            # Repeat frames if video is too short
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

        frames = vr.get_batch(indices).asnumpy()
        return frames

    except ImportError:
        logger.warning("decord not available, falling back to PyAV")
        return sample_frames_pyav(video_path, num_frames, strategy)

def sample_frames_pyav(
    video_path: Union[str, Path],
    num_frames: int,
    strategy: str = "uniform"
) -> NDArray[np.uint8]:
    """
    Sample frames from video using PyAV (fallback).

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        strategy: Sampling strategy.

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
    Find all frame directories matching the expected structure.

    Args:
        base_dir: Base directory to search from.
        pattern: Glob pattern to match frame directories.

    Returns:
        List of paths to frame directories.
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
    Optimized directory search using os.scandir for specific structure.
    
    Structure: trainlake/[yy].[mm]w/N[7digits]-[12digits]/RAW_DB/*_CMR*/CMR_GT_Frame/
    """
    base_dir = Path(base_dir)
    results = []
    dirs_scanned = 0
    
    # Check if base_dir exists
    if not base_dir.exists():
        logger.warning(f"Base directory {base_dir} does not exist")
        return []

    # 1. Week directories ([yy].[mm]w)
    for week_dir in base_dir.iterdir():
        if not week_dir.is_dir():
            continue
        if not re.match(r'\d{2}\.\d{2}w', week_dir.name):
            continue

        # 2. Video directories (N[7digits]-[12digits])
        for video_dir in week_dir.iterdir():
            if not video_dir.is_dir():
                continue
            if not re.match(r'^N\d{7}-\d{12}$', video_dir.name):
                continue
            
            dirs_scanned += 1
            if progress_callback:
                progress_callback(video_dir.name, dirs_scanned)
                
            # 3. RAW_DB/*_CMR*/CMR_GT_Frame
            raw_db = video_dir / "RAW_DB"
            if raw_db.exists():
                for cmr_dir in raw_db.iterdir():
                    if cmr_dir.is_dir() and "_CMR" in cmr_dir.name:
                        frame_dir = cmr_dir / target_name
                        if frame_dir.exists():
                            results.append(frame_dir)
                            
    return results

def find_frame_directories_cached(
    base_dir: Union[str, Path],
    cache_file: Optional[Union[str, Path]] = None,
    force_rescan: bool = False,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> list[Path]:
    """
    Find frame directories with caching support.
    """
    base_dir = Path(base_dir)
    if cache_file and cache_file.exists() and not force_rescan:
        logger.info(f"Loading frame directories from cache: {cache_file}")
        try:
            content = cache_file.read_text()
            paths = [Path(p) for p in content.splitlines() if p.strip()]
            return paths
        except Exception as e:
            logger.warning(f"Failed to read cache file: {e}")
    
    # Rescan
    results = find_frame_directories_optimized(
        base_dir, 
        progress_callback=progress_callback
    )
    
    # Save cache
    if cache_file and results:
        try:
            cache_path = Path(cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("\n".join(str(p) for p in results))
            logger.info(f"Saved {len(results)} directories to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache file: {e}")
            
    return results