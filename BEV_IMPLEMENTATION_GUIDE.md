# 멀티 카메라 BEV 통합 구현 가이드

## 개요

**목표**: 11대 자율주행 카메라 영상을 IPM(Inverse Perspective Mapping)으로 BEV(Bird's Eye View) 이미지로 변환하여 비디오당 단일 임베딩 생성

**현재 문제**:
- `pipeline/embedder/frame_loader.py`의 `find_frame_directories_optimized()`가 각 카메라를 독립 비디오로 처리
- 같은 비디오의 11개 카메라가 모두 `N1234567-240115120000.npy`로 저장되어 마지막 카메라만 남음
- 멀티 카메라의 공간적 관계가 완전히 무시됨

---

## 카메라 구성 (11대)

| 위치 | 카메라 | 디렉토리 패턴 예시 |
|-----|-------|------------------|
| 전방 | 원거리 | `*FRONT*LONG*_CMR` |
| 전방 | 협각 | `*FRONT*NARROW*_CMR` |
| 전방 | 광각 | `*FRONT*WIDE*_CMR` |
| 좌측 | 전방대각선 | `*LEFT*FRONT*_CMR` |
| 좌측 | 후방대각선 | `*LEFT*REAR*_CMR` |
| 좌측 | 광각 | `*LEFT*WIDE*_CMR` |
| 우측 | 전방대각선 | `*RIGHT*FRONT*_CMR` |
| 우측 | 후방대각선 | `*RIGHT*REAR*_CMR` |
| 우측 | 광각 | `*RIGHT*WIDE*_CMR` |
| 후방 | 협각 | `*REAR*NARROW*_CMR` |
| 후방 | 광각 | `*REAR*WIDE*_CMR` |

---

## 디렉토리 구조

### 입력 데이터 구조
```
trainlake/
└── 24.01w/
    └── N1234567-240115120000/
        └── RAW_DB/
            ├── FRONT_LONG_CMR/
            │   └── CMR_GT_Frame/
            │       ├── frame_0000.jpg
            │       ├── frame_0001.jpg
            │       └── ...
            ├── FRONT_NARROW_CMR/
            │   └── CMR_GT_Frame/
            │       └── ...
            ├── LEFT_FRONT_CMR/
            │   └── CMR_GT_Frame/
            │       └── ...
            └── ... (11개 카메라)
```

### 캘리브레이션 데이터 구조
```
calibrations/
└── N1234567/
    └── calibration.json (또는 .yaml)
```

### 캘리브레이션 파일 형식 (예상)
```json
{
  "cameras": {
    "FRONT_LONG_CMR": {
      "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "distortion": [k1, k2, p1, p2, k3],
      "rotation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
      "translation": [tx, ty, tz],
      "width": 1920,
      "height": 1080,
      "fisheye": false
    },
    "LEFT_WIDE_CMR": {
      "intrinsic": [...],
      "distortion": [k1, k2, k3, k4],
      "rotation": [...],
      "translation": [...],
      "width": 1920,
      "height": 1080,
      "fisheye": true
    }
  }
}
```

**참고**: extrinsic (rotation, translation)은 **카메라 → 차량 중심** 좌표계 변환

---

## 새로 생성할 파일

### 1. `pipeline/bev/__init__.py`

```python
"""BEV (Bird's Eye View) generation package."""

from .config import BEVConfig
from .calibration import CameraCalibration, CalibrationLoader
from .ipm import IPMTransformer
from .stitcher import BEVStitcher
from .generator import BEVGenerator

__all__ = [
    "BEVConfig",
    "CameraCalibration",
    "CalibrationLoader",
    "IPMTransformer",
    "BEVStitcher",
    "BEVGenerator",
]
```

---

### 2. `pipeline/bev/config.py`

```python
"""BEV generation configuration."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BEVConfig:
    """Configuration for BEV generation."""

    # Output BEV image dimensions (pixels)
    bev_width: int = 800
    bev_height: int = 800

    # Physical BEV coverage (meters from vehicle center)
    # x: left(-) / right(+), y: rear(-) / front(+)
    bev_x_range: Tuple[float, float] = (-25.0, 25.0)
    bev_y_range: Tuple[float, float] = (-25.0, 25.0)

    # IPM ground plane assumption (meters)
    ground_height: float = 0.0

    # Blending parameters for overlapping regions
    blend_overlap_pixels: int = 50
    use_weighted_blend: bool = True

    # Calibration settings
    calibration_dir: str = ""
    calibration_format: str = "json"  # "json" or "yaml"

    # Minimum cameras required (graceful degradation if fewer)
    min_cameras: int = 4
```

---

### 3. `pipeline/bev/calibration.py`

```python
"""Camera calibration loading and management."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """Single camera calibration parameters."""

    camera_id: str

    # Intrinsic parameters (3x3 matrix)
    # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic_matrix: np.ndarray

    # Distortion coefficients
    # Standard: [k1, k2, p1, p2, k3] (5 params)
    # Fisheye: [k1, k2, k3, k4] (4 params)
    distortion_coeffs: np.ndarray

    # Extrinsic: Camera to Vehicle Center transformation
    rotation_matrix: np.ndarray     # 3x3
    translation_vector: np.ndarray  # 3x1 (meters)

    # Image dimensions
    image_width: int
    image_height: int

    # Camera type
    is_fisheye: bool = False

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """4x4 extrinsic transformation matrix [R|t]."""
        ext = np.eye(4)
        ext[:3, :3] = self.rotation_matrix
        ext[:3, 3] = self.translation_vector.flatten()
        return ext

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 projection matrix P = K @ [R|t]."""
        rt = np.hstack([self.rotation_matrix, self.translation_vector.reshape(3, 1)])
        return self.intrinsic_matrix @ rt


class CalibrationLoader:
    """Load and manage multi-camera calibration data."""

    def __init__(self, calibration_base_dir: Path):
        """
        Initialize calibration loader.

        Args:
            calibration_base_dir: Directory containing N[7digits]/ subdirectories
        """
        self.calibration_base_dir = Path(calibration_base_dir)
        self._cache: Dict[str, Dict[str, CameraCalibration]] = {}

    def load_calibration(self, vehicle_id: str) -> Dict[str, CameraCalibration]:
        """
        Load calibration for a vehicle.

        Args:
            vehicle_id: Vehicle ID in N[7digits] format (e.g., "N1234567")

        Returns:
            Dictionary mapping camera_id to CameraCalibration

        Raises:
            FileNotFoundError: If calibration file not found
        """
        if vehicle_id in self._cache:
            return self._cache[vehicle_id]

        calib_dir = self.calibration_base_dir / vehicle_id
        calib_file = self._find_calibration_file(calib_dir)

        if calib_file is None:
            raise FileNotFoundError(
                f"No calibration file found for vehicle {vehicle_id} in {calib_dir}"
            )

        calibrations = self._parse_calibration_file(calib_file)
        self._cache[vehicle_id] = calibrations

        logger.info(f"Loaded {len(calibrations)} camera calibrations for {vehicle_id}")
        return calibrations

    def _find_calibration_file(self, calib_dir: Path) -> Optional[Path]:
        """Find calibration file (JSON or YAML) in directory."""
        if not calib_dir.exists():
            return None

        # Try common naming conventions
        for name in ["calibration", "calib", "camera_params", "cameras"]:
            for ext in [".json", ".yaml", ".yml"]:
                path = calib_dir / f"{name}{ext}"
                if path.exists():
                    return path

        # Try any JSON/YAML file
        for ext in [".json", ".yaml", ".yml"]:
            files = list(calib_dir.glob(f"*{ext}"))
            if files:
                return files[0]

        return None

    def _parse_calibration_file(self, path: Path) -> Dict[str, CameraCalibration]:
        """Parse calibration file into CameraCalibration objects."""
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML calibration files")

        calibrations = {}

        # Handle different possible structures
        cameras_data = data.get("cameras", data)

        for camera_id, params in cameras_data.items():
            try:
                calibrations[camera_id] = CameraCalibration(
                    camera_id=camera_id,
                    intrinsic_matrix=np.array(params["intrinsic"], dtype=np.float64),
                    distortion_coeffs=np.array(params["distortion"], dtype=np.float64),
                    rotation_matrix=np.array(params["rotation"], dtype=np.float64),
                    translation_vector=np.array(params["translation"], dtype=np.float64),
                    image_width=int(params.get("width", 1920)),
                    image_height=int(params.get("height", 1080)),
                    is_fisheye=bool(params.get("fisheye", False))
                )
            except KeyError as e:
                logger.warning(f"Skipping camera {camera_id}: missing key {e}")
                continue

        return calibrations
```

---

### 4. `pipeline/bev/ipm.py`

```python
"""Inverse Perspective Mapping (IPM) transformation."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from .calibration import CameraCalibration

logger = logging.getLogger(__name__)


class IPMTransformer:
    """
    Inverse Perspective Mapping transformer.

    Transforms camera images to Bird's Eye View using homography.
    Assumes flat ground plane.

    References:
    - https://gaussian37.github.io/vision-concept-ipm/
    - https://github.com/DrMahdiRezaei/Birds-Eye-View-Calibration
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        bev_width: int = 800,
        bev_height: int = 800,
        x_range: Tuple[float, float] = (-25.0, 25.0),
        y_range: Tuple[float, float] = (-25.0, 25.0),
        ground_height: float = 0.0
    ):
        """
        Initialize IPM transformer.

        Args:
            calibration: Camera calibration parameters
            bev_width: Output BEV image width in pixels
            bev_height: Output BEV image height in pixels
            x_range: BEV X range in meters (left/right from vehicle center)
            y_range: BEV Y range in meters (rear/front from vehicle center)
            ground_height: Ground plane height in meters (usually 0)
        """
        self.calibration = calibration
        self.bev_width = bev_width
        self.bev_height = bev_height
        self.x_range = x_range
        self.y_range = y_range
        self.ground_height = ground_height

        # Precompute transformation
        self._compute_homography()
        self._compute_remap_tables()

    def _compute_homography(self) -> None:
        """Compute IPM homography matrix."""
        # Scale factors: meters to pixels
        self.scale_x = self.bev_width / (self.x_range[1] - self.x_range[0])
        self.scale_y = self.bev_height / (self.y_range[1] - self.y_range[0])

        # BEV pixel (u, v) -> world (x, y, ground_height) transformation
        # BEV image: (0,0) is top-left, y increases downward
        # World: x is right, y is forward
        self.bev_to_world = np.array([
            [1.0 / self.scale_x, 0, self.x_range[0]],
            [0, -1.0 / self.scale_y, self.y_range[1]],  # flip y-axis
            [0, 0, 1]
        ], dtype=np.float64)

        # Camera projection: world (x, y, z=ground_height) -> image (u, v)
        K = self.calibration.intrinsic_matrix
        R = self.calibration.rotation_matrix
        t = self.calibration.translation_vector.reshape(3, 1)

        # For ground plane z=ground_height:
        # P_ground = K @ [r1, r2, r3*z + t] where r1, r2, r3 are columns of R
        r1 = R[:, 0:1]
        r2 = R[:, 1:2]
        r3 = R[:, 2:3]

        H_world_to_image = K @ np.hstack([r1, r2, r3 * self.ground_height + t])

        # Complete homography: BEV pixel -> Image pixel
        self.H_bev_to_image = H_world_to_image @ self.bev_to_world

        # Inverse: Image pixel -> BEV pixel (for reference)
        try:
            self.H_image_to_bev = np.linalg.inv(self.H_bev_to_image)
        except np.linalg.LinAlgError:
            logger.warning(f"Singular homography for camera {self.calibration.camera_id}")
            self.H_image_to_bev = None

    def _compute_remap_tables(self) -> None:
        """Precompute OpenCV remap tables for fast transformation."""
        # Generate BEV coordinate grid
        u_bev = np.arange(self.bev_width)
        v_bev = np.arange(self.bev_height)
        u_grid, v_grid = np.meshgrid(u_bev, v_bev)

        # Homogeneous coordinates
        bev_coords = np.stack([
            u_grid.flatten(),
            v_grid.flatten(),
            np.ones(self.bev_width * self.bev_height)
        ], axis=0)  # Shape: (3, H*W)

        # Transform to image coordinates
        img_coords = self.H_bev_to_image @ bev_coords

        # Normalize homogeneous coordinates
        img_coords = img_coords / (img_coords[2:3] + 1e-10)

        # Reshape to remap format
        self.map_x = img_coords[0].reshape(self.bev_height, self.bev_width).astype(np.float32)
        self.map_y = img_coords[1].reshape(self.bev_height, self.bev_width).astype(np.float32)

        # Create validity mask (points within original image bounds)
        valid_x = (self.map_x >= 0) & (self.map_x < self.calibration.image_width - 1)
        valid_y = (self.map_y >= 0) & (self.map_y < self.calibration.image_height - 1)
        self.valid_mask = (valid_x & valid_y).astype(np.uint8)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from image.

        Args:
            image: Input distorted image (H, W, C)

        Returns:
            Undistorted image
        """
        K = self.calibration.intrinsic_matrix
        dist = self.calibration.distortion_coeffs

        if self.calibration.is_fisheye:
            # Fisheye undistortion (4 distortion params: k1, k2, k3, k4)
            return cv2.fisheye.undistortImage(
                image, K, dist[:4], Knew=K
            )
        else:
            # Standard undistortion (5 params: k1, k2, p1, p2, k3)
            return cv2.undistort(image, K, dist)

    def transform(
        self,
        image: np.ndarray,
        undistort: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform camera image to BEV.

        Args:
            image: Input camera image (H, W, C) in BGR or RGB
            undistort: Whether to remove lens distortion first

        Returns:
            Tuple of:
                - bev_image: BEV transformed image (bev_height, bev_width, C)
                - valid_mask: Binary mask of valid pixels (bev_height, bev_width)
        """
        if undistort:
            image = self.undistort_image(image)

        # Apply IPM transformation using precomputed remap tables
        bev_image = cv2.remap(
            image,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return bev_image, self.valid_mask.copy()
```

---

### 5. `pipeline/bev/stitcher.py`

```python
"""Multi-camera BEV stitching with weighted blending."""

from __future__ import annotations

import logging
from typing import Dict

import cv2
import numpy as np

from .config import BEVConfig
from .calibration import CameraCalibration
from .ipm import IPMTransformer

logger = logging.getLogger(__name__)


class BEVStitcher:
    """
    Stitch multiple camera BEV views into unified surround view.

    Uses distance-based weighted blending in overlapping regions.
    """

    def __init__(
        self,
        config: BEVConfig,
        calibrations: Dict[str, CameraCalibration]
    ):
        """
        Initialize BEV stitcher.

        Args:
            config: BEV configuration
            calibrations: Dictionary mapping camera_id to CameraCalibration
        """
        self.config = config
        self.calibrations = calibrations

        # Initialize IPM transformers for each camera
        self.transformers: Dict[str, IPMTransformer] = {}
        self._init_transformers()

        # Precompute blending weights
        self.blend_weights: Dict[str, np.ndarray] = {}
        self._compute_blend_weights()

    def _init_transformers(self) -> None:
        """Initialize IPM transformer for each camera."""
        for camera_id, calib in self.calibrations.items():
            try:
                self.transformers[camera_id] = IPMTransformer(
                    calibration=calib,
                    bev_width=self.config.bev_width,
                    bev_height=self.config.bev_height,
                    x_range=self.config.bev_x_range,
                    y_range=self.config.bev_y_range,
                    ground_height=self.config.ground_height
                )
                logger.debug(f"Initialized IPM transformer for {camera_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize transformer for {camera_id}: {e}")

    def _compute_blend_weights(self) -> None:
        """
        Precompute blending weights for each camera.

        Creates distance-based weights that smoothly blend overlapping regions.
        """
        for camera_id, transformer in self.transformers.items():
            # Start with valid mask as base weight
            mask = transformer.valid_mask.astype(np.float32)

            if self.config.use_weighted_blend:
                # Compute distance transform from invalid regions
                # This gives higher weight to pixels far from boundaries
                dist = cv2.distanceTransform(
                    mask.astype(np.uint8),
                    cv2.DIST_L2,
                    5
                )

                # Normalize and apply smooth falloff
                if dist.max() > 0:
                    dist = dist / dist.max()

                # Apply sigmoid-like falloff for smoother blending
                weight = np.clip(dist * 2, 0, 1) * mask
            else:
                weight = mask

            self.blend_weights[camera_id] = weight

        # Normalize weights so they sum to 1 at each pixel
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize blending weights to sum to 1 at each pixel."""
        weight_sum = np.zeros(
            (self.config.bev_height, self.config.bev_width),
            dtype=np.float32
        )

        for weight in self.blend_weights.values():
            weight_sum += weight

        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-6)

        for camera_id in self.blend_weights:
            self.blend_weights[camera_id] = self.blend_weights[camera_id] / weight_sum

    def stitch(
        self,
        images: Dict[str, np.ndarray],
        undistort: bool = True
    ) -> np.ndarray:
        """
        Stitch multiple camera images into unified BEV.

        Args:
            images: Dictionary mapping camera_id to image array (H, W, 3)
                    Images should be in RGB format
            undistort: Whether to undistort images first

        Returns:
            Unified BEV image (bev_height, bev_width, 3) in RGB format
        """
        # Initialize output accumulator
        bev_result = np.zeros(
            (self.config.bev_height, self.config.bev_width, 3),
            dtype=np.float32
        )

        cameras_used = 0

        for camera_id, image in images.items():
            if camera_id not in self.transformers:
                logger.debug(f"No transformer for camera {camera_id}, skipping")
                continue

            if camera_id not in self.blend_weights:
                logger.debug(f"No blend weight for camera {camera_id}, skipping")
                continue

            try:
                # Transform to BEV
                transformer = self.transformers[camera_id]
                bev_view, _ = transformer.transform(image, undistort=undistort)

                # Get blending weight (expand to 3 channels)
                weight = self.blend_weights[camera_id]
                weight_3ch = weight[:, :, np.newaxis]

                # Accumulate weighted contribution
                bev_result += bev_view.astype(np.float32) * weight_3ch
                cameras_used += 1

            except Exception as e:
                logger.warning(f"Failed to process camera {camera_id}: {e}")
                continue

        logger.debug(f"Stitched BEV from {cameras_used} cameras")

        # Clip and convert to uint8
        return np.clip(bev_result, 0, 255).astype(np.uint8)

    def get_coverage_mask(self) -> np.ndarray:
        """
        Get mask showing which pixels have camera coverage.

        Returns:
            Binary mask (bev_height, bev_width) where 255 = covered
        """
        coverage = np.zeros(
            (self.config.bev_height, self.config.bev_width),
            dtype=np.float32
        )

        for weight in self.blend_weights.values():
            coverage = np.maximum(coverage, weight)

        return (coverage > 0.01).astype(np.uint8) * 255
```

---

### 6. `pipeline/bev/generator.py`

```python
"""BEV generation orchestrator."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .config import BEVConfig
from .calibration import CalibrationLoader
from .stitcher import BEVStitcher

logger = logging.getLogger(__name__)


class BEVGenerator:
    """
    Main BEV generation orchestrator.

    Coordinates calibration loading, multi-camera frame loading,
    and BEV generation.
    """

    def __init__(
        self,
        config: Optional[BEVConfig] = None,
        calibration_dir: Optional[Path] = None
    ):
        """
        Initialize BEV generator.

        Args:
            config: BEV configuration
            calibration_dir: Directory containing calibration files
        """
        self.config = config or BEVConfig()

        calib_dir = calibration_dir or Path(self.config.calibration_dir)
        self.calibration_loader = CalibrationLoader(calib_dir)

        # Cache stitchers per vehicle (different vehicles may have different calibrations)
        self._stitcher_cache: Dict[str, BEVStitcher] = {}

    def _get_stitcher(self, vehicle_id: str) -> BEVStitcher:
        """Get or create stitcher for a vehicle."""
        if vehicle_id not in self._stitcher_cache:
            calibrations = self.calibration_loader.load_calibration(vehicle_id)
            self._stitcher_cache[vehicle_id] = BEVStitcher(self.config, calibrations)
            logger.info(f"Created BEV stitcher for vehicle {vehicle_id}")
        return self._stitcher_cache[vehicle_id]

    def _extract_vehicle_id(self, video_dir: Path) -> str:
        """
        Extract vehicle ID (N-number) from video directory path.

        Args:
            video_dir: Path containing N1234567-YYMMDDhhmmss pattern

        Returns:
            Vehicle ID (e.g., "N1234567")
        """
        # Pattern: N[7 digits]-[12 digits timestamp]
        pattern = re.compile(r'(N\d{7})-\d{12}')

        for part in video_dir.parts:
            match = pattern.match(part)
            if match:
                return match.group(1)

        # Fallback: try the directory name itself
        match = pattern.match(video_dir.name)
        if match:
            return match.group(1)

        raise ValueError(f"Could not extract vehicle ID from {video_dir}")

    def generate_bev_sequence(
        self,
        video_dir: Path,
        camera_frame_dirs: Dict[str, Path],
        frame_indices: Optional[List[int]] = None,
        num_frames: int = 16
    ) -> List[np.ndarray]:
        """
        Generate BEV sequence for a video.

        Args:
            video_dir: Path to video directory (contains N-number in path)
            camera_frame_dirs: Mapping of camera_id to CMR_GT_Frame directory
            frame_indices: Specific frame indices to process (None = uniform sampling)
            num_frames: Number of frames to sample if frame_indices is None

        Returns:
            List of BEV images (one per sampled frame), each (H, W, 3) RGB
        """
        # Get stitcher for this vehicle
        vehicle_id = self._extract_vehicle_id(video_dir)
        stitcher = self._get_stitcher(vehicle_id)

        # Get frame files for each camera
        camera_frames: Dict[str, List[Path]] = {}
        min_frame_count = float('inf')

        for camera_id, frame_dir in camera_frame_dirs.items():
            frames = sorted(frame_dir.glob("*.jpg"))
            if not frames:
                frames = sorted(frame_dir.glob("*.JPG"))
            if not frames:
                frames = sorted(frame_dir.glob("*.png"))

            if frames:
                camera_frames[camera_id] = frames
                min_frame_count = min(min_frame_count, len(frames))

        if not camera_frames:
            raise ValueError(f"No frames found in any camera directory")

        if min_frame_count == float('inf'):
            min_frame_count = 0

        logger.info(f"Found {len(camera_frames)} cameras with {min_frame_count} min frames")

        # Determine frame indices to sample
        if frame_indices is None:
            if min_frame_count <= num_frames:
                frame_indices = list(range(int(min_frame_count)))
            else:
                frame_indices = np.linspace(
                    0, min_frame_count - 1, num_frames, dtype=int
                ).tolist()

        # Generate BEV for each frame
        bev_images = []

        for frame_idx in frame_indices:
            # Load images from all cameras at this timestamp
            images: Dict[str, np.ndarray] = {}

            for camera_id, frames in camera_frames.items():
                if frame_idx < len(frames):
                    img = Image.open(frames[frame_idx]).convert('RGB')
                    images[camera_id] = np.array(img)

            if len(images) < self.config.min_cameras:
                logger.warning(
                    f"Frame {frame_idx}: only {len(images)} cameras available "
                    f"(min required: {self.config.min_cameras})"
                )

            # Stitch to BEV
            bev = stitcher.stitch(images)
            bev_images.append(bev)

        return bev_images

    def generate_single_bev(
        self,
        video_dir: Path,
        camera_images: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Generate single BEV from camera images.

        Args:
            video_dir: Path to video directory
            camera_images: Dictionary mapping camera_id to image array (RGB)

        Returns:
            BEV image (H, W, 3) RGB
        """
        vehicle_id = self._extract_vehicle_id(video_dir)
        stitcher = self._get_stitcher(vehicle_id)
        return stitcher.stitch(camera_images)
```

---

### 7. `pipeline/embedder/multicam_loader.py`

```python
"""Multi-camera video discovery and frame loading."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MultiCameraVideoInfo:
    """Information about a multi-camera video."""

    video_name: str                    # N1234567-YYMMDDhhmmss
    video_dir: Path                    # Full path to video directory
    camera_dirs: Dict[str, Path]       # camera_id -> CMR_GT_Frame path
    frame_count: int                   # Minimum frames across all cameras


def find_multicam_videos(
    base_dir: Path,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> List[MultiCameraVideoInfo]:
    """
    Find all multi-camera videos and group their camera directories.

    Expected structure:
    base_dir/
    └── [yy].[mm]w/                      # Week directory (e.g., 24.01w)
        └── N[7digits]-[12digits]/       # Video directory
            └── RAW_DB/
                ├── *_CMR/               # Camera directories
                │   └── CMR_GT_Frame/    # Frame directory
                └── ...

    Args:
        base_dir: Base directory to search
        progress_callback: Optional callback(video_name, count) for progress

    Returns:
        List of MultiCameraVideoInfo objects, one per video
    """
    base_dir = Path(base_dir)
    results = []

    # Patterns
    week_pattern = re.compile(r'^\d{2}\.\d{2}w$')           # 24.01w
    video_pattern = re.compile(r'^(N\d{7})-(\d{12})$')      # N1234567-240115120000

    videos_scanned = 0

    for week_entry in sorted(base_dir.iterdir()):
        if not week_entry.is_dir():
            continue
        if not week_pattern.match(week_entry.name):
            continue

        for video_entry in sorted(week_entry.iterdir()):
            if not video_entry.is_dir():
                continue

            match = video_pattern.match(video_entry.name)
            if not match:
                continue

            videos_scanned += 1
            if progress_callback:
                progress_callback(video_entry.name, videos_scanned)

            # Look for RAW_DB directory
            raw_db_dir = video_entry / "RAW_DB"
            if not raw_db_dir.exists():
                logger.debug(f"No RAW_DB in {video_entry}")
                continue

            # Find all camera directories
            camera_dirs: Dict[str, Path] = {}
            min_frames = float('inf')

            for cmr_entry in sorted(raw_db_dir.iterdir()):
                if not cmr_entry.is_dir():
                    continue
                if "_CMR" not in cmr_entry.name:
                    continue

                frame_dir = cmr_entry / "CMR_GT_Frame"
                if not frame_dir.exists():
                    continue

                # Count frames
                frame_count = len(list(frame_dir.glob("*.jpg")))
                if frame_count == 0:
                    frame_count = len(list(frame_dir.glob("*.JPG")))
                if frame_count == 0:
                    frame_count = len(list(frame_dir.glob("*.png")))

                if frame_count > 0:
                    camera_id = cmr_entry.name  # e.g., "FRONT_LONG_CMR"
                    camera_dirs[camera_id] = frame_dir
                    min_frames = min(min_frames, frame_count)

            # Only include if we have at least one camera
            if camera_dirs:
                results.append(MultiCameraVideoInfo(
                    video_name=video_entry.name,
                    video_dir=video_entry,
                    camera_dirs=camera_dirs,
                    frame_count=int(min_frames) if min_frames != float('inf') else 0
                ))

    logger.info(
        f"Found {len(results)} multi-camera videos "
        f"(total {sum(len(v.camera_dirs) for v in results)} camera directories)"
    )

    return results


def load_multicam_frames_for_internvl(
    video_info: MultiCameraVideoInfo,
    bev_generator,  # BEVGenerator instance
    transform: Callable,
    num_frames: int = 16,
    strategy: str = "uniform"
) -> Tuple[torch.Tensor, List[int]]:
    """
    Load multi-camera frames as BEV and prepare for InternVL.

    Args:
        video_info: Multi-camera video information
        bev_generator: BEVGenerator instance for BEV generation
        transform: InternVL image transform function
        num_frames: Number of frames to sample
        strategy: Sampling strategy ("uniform" only currently)

    Returns:
        Tuple of:
            - pixel_values: Tensor of shape (num_frames, C, H, W)
            - num_patches_list: List of patch counts per frame
    """
    # Generate BEV sequence
    bev_images = bev_generator.generate_bev_sequence(
        video_dir=video_info.video_dir,
        camera_frame_dirs=video_info.camera_dirs,
        num_frames=num_frames
    )

    if not bev_images:
        raise ValueError(f"No BEV images generated for {video_info.video_name}")

    pixel_values_list = []
    num_patches_list = []

    for bev_image in bev_images:
        # Convert numpy array to PIL Image for transform
        pil_image = Image.fromarray(bev_image)

        # Apply InternVL transform
        pixel_values = transform(pil_image)
        pixel_values = pixel_values.unsqueeze(0)  # Add batch dim: (1, C, H, W)

        pixel_values_list.append(pixel_values)
        num_patches_list.append(1)

    # Concatenate all frames
    pixel_values = torch.cat(pixel_values_list, dim=0)  # (num_frames, C, H, W)
    pixel_values = pixel_values.to(torch.bfloat16)

    return pixel_values, num_patches_list
```

---

### 8. `pipeline/embedder/bev_embedder.py`

```python
"""BEV-based embedding extraction for multi-camera videos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray

from ..config import VideoEmbedderConfig
from ..bev.config import BEVConfig
from ..bev.generator import BEVGenerator
from .multicam_loader import (
    MultiCameraVideoInfo,
    find_multicam_videos,
    load_multicam_frames_for_internvl,
)
from . import model_loader

logger = logging.getLogger(__name__)


class BEVEmbedder:
    """
    Extract embeddings from multi-camera videos using BEV representation.

    Pipeline:
    1. Group 11 cameras by video (N-number + timestamp)
    2. Generate unified BEV from IPM transformation
    3. Extract embeddings using InternVL vision encoder
    4. Save one embedding per video
    """

    def __init__(
        self,
        embedder_config: Optional[VideoEmbedderConfig] = None,
        bev_config: Optional[BEVConfig] = None,
        calibration_dir: Optional[Path] = None
    ):
        """
        Initialize BEV embedder.

        Args:
            embedder_config: Video embedder configuration
            bev_config: BEV generation configuration
            calibration_dir: Directory containing calibration files
        """
        self.embedder_config = embedder_config or VideoEmbedderConfig()
        self.bev_config = bev_config or BEVConfig()

        # Initialize BEV generator
        self.bev_generator = BEVGenerator(
            config=self.bev_config,
            calibration_dir=calibration_dir
        )

        # Model state (lazy loading)
        self.device = None
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self._transform = None
        self._model_loader = None

    def load_model(self) -> None:
        """Load InternVL model for embedding extraction."""
        if self._is_loaded:
            return

        self.device = torch.device(self.embedder_config.device)

        self._model_loader = model_loader.InternVLModelLoader(
            model_name=self.embedder_config.model_name,
            input_size=self.embedder_config.input_size,
            torch_dtype=self.embedder_config.torch_dtype,
            use_flash_attn=self.embedder_config.use_flash_attn,
            trust_remote_code=self.embedder_config.trust_remote_code,
            device=str(self.device)
        )

        self.model, self.tokenizer, self._transform = self._model_loader.load_model()
        self._is_loaded = True

        logger.info(f"Loaded InternVL model on {self.device}")

    def extract_embedding_from_multicam(
        self,
        video_info: MultiCameraVideoInfo,
        save_embedding: bool = True
    ) -> Tuple[NDArray[np.float32], str]:
        """
        Extract embedding from multi-camera video using BEV.

        Args:
            video_info: Multi-camera video information
            save_embedding: Whether to save the embedding to disk

        Returns:
            Tuple of (embedding vector (4096,), video name)
        """
        if not self._is_loaded:
            self.load_model()

        try:
            # Load BEV frames with InternVL preprocessing
            pixel_values, num_patches_list = load_multicam_frames_for_internvl(
                video_info=video_info,
                bev_generator=self.bev_generator,
                transform=self._transform,
                num_frames=self.embedder_config.num_frames,
                strategy=self.embedder_config.frame_sample_strategy
            )

            # Move to device
            pixel_values = pixel_values.to(self.device)

            # Extract vision features using InternVL
            with torch.no_grad():
                vit_embeds = self._model_loader.extract_vision_features(
                    self.model, pixel_values
                )

            # Mean pooling across frames and patches
            # vit_embeds shape: (num_frames, num_patches, hidden_dim=4096)
            embedding = vit_embeds.mean(dim=(0, 1))  # (4096,)

            # L2 normalize
            if self.embedder_config.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

            # Convert to numpy
            embedding_np = embedding.to(torch.float32).cpu().numpy().astype(np.float32)

            # Save
            if save_embedding:
                self._save_embedding(embedding_np, video_info.video_name)

            return embedding_np, video_info.video_name

        except Exception as e:
            logger.error(f"Failed to process {video_info.video_name}: {e}")
            raise RuntimeError(
                f"Failed to extract BEV embedding from {video_info.video_name}: {e}"
            ) from e

    def _save_embedding(
        self,
        embedding: NDArray[np.float32],
        video_name: str
    ) -> Path:
        """Save embedding to output directory."""
        output_dir = Path(self.embedder_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # video_name is like "N1234567-240115120000"
        embedding_filename = f"{video_name}.npy"
        output_path = output_dir / embedding_filename

        np.save(output_path, embedding)
        logger.debug(f"Saved BEV embedding to {output_path}")

        return output_path

    def extract_embeddings_from_base_dir(
        self,
        base_dir: Path,
        save_embeddings: bool = True,
        show_progress: bool = True
    ) -> Tuple[NDArray[np.float32], List[str], List[int]]:
        """
        Extract embeddings from all multi-camera videos in base directory.

        Args:
            base_dir: Base directory containing video data
            save_embeddings: Whether to save embeddings to disk
            show_progress: Whether to show progress bar

        Returns:
            Tuple of:
                - embeddings: Array of shape (num_videos, 4096)
                - video_names: List of video names
                - failed_indices: Indices of videos that failed processing
        """
        if not self._is_loaded:
            self.load_model()

        # Find all multi-camera videos
        print("Scanning for multi-camera videos...")

        def progress_callback(name: str, count: int):
            print(f"\rScanning: {count} videos | {name[:40]:<40}", end="", flush=True)

        video_infos = find_multicam_videos(base_dir, progress_callback=progress_callback)
        print(f"\nFound {len(video_infos)} multi-camera videos")

        if not video_infos:
            raise RuntimeError(f"No multi-camera videos found in {base_dir}")

        embeddings = []
        video_names = []
        failed_indices = []

        # Process each video
        iterator = enumerate(video_infos)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    list(iterator),
                    desc="Extracting BEV embeddings",
                    unit="video"
                )
            except ImportError:
                pass

        for idx, video_info in iterator:
            try:
                embedding, video_name = self.extract_embedding_from_multicam(
                    video_info, save_embedding=save_embeddings
                )
                embeddings.append(embedding)
                video_names.append(video_name)

            except Exception as e:
                logger.warning(f"Failed to process {video_info.video_name}: {e}")
                failed_indices.append(idx)

        if not embeddings:
            raise RuntimeError("No videos could be processed successfully")

        logger.info(
            f"Successfully processed {len(embeddings)} videos, "
            f"{len(failed_indices)} failed"
        )

        return np.stack(embeddings), video_names, failed_indices
```

---

## 수정할 기존 파일

### 9. `pipeline/cli.py` 수정

**추가할 import:**
```python
from .bev.config import BEVConfig
from .embedder.bev_embedder import BEVEmbedder
```

**추가할 argparse 옵션 (기존 옵션 뒤에 추가):**
```python
# BEV mode options
parser.add_argument(
    "--bev-mode",
    action="store_true",
    help="Enable BEV mode for multi-camera processing"
)
parser.add_argument(
    "--calibration-dir",
    type=str,
    default=None,
    help="Directory containing camera calibration files (N[7digits]/ subdirs)"
)
parser.add_argument(
    "--bev-width",
    type=int,
    default=800,
    help="BEV output width in pixels (default: 800)"
)
parser.add_argument(
    "--bev-height",
    type=int,
    default=800,
    help="BEV output height in pixels (default: 800)"
)
parser.add_argument(
    "--bev-range",
    type=float,
    default=25.0,
    help="BEV coverage range in meters from vehicle center (default: 25.0)"
)
```

**추가할 처리 로직 (frame_dir 처리 부분에서):**
```python
if args.frame_dir:
    if args.bev_mode:
        # BEV mode: multi-camera processing
        if not args.calibration_dir:
            raise ValueError("--calibration-dir required for BEV mode")

        bev_config = BEVConfig(
            bev_width=args.bev_width,
            bev_height=args.bev_height,
            bev_x_range=(-args.bev_range, args.bev_range),
            bev_y_range=(-args.bev_range, args.bev_range),
            calibration_dir=args.calibration_dir,
        )

        bev_embedder = BEVEmbedder(
            embedder_config=VideoEmbedderConfig(
                output_dir=args.output_dir or "output",
                num_frames=args.num_frames or 16,
            ),
            bev_config=bev_config,
            calibration_dir=Path(args.calibration_dir),
        )

        embeddings, video_names, failed = bev_embedder.extract_embeddings_from_base_dir(
            Path(args.frame_dir),
            save_embeddings=True,
            show_progress=True,
        )

        print(f"\nExtracted {len(video_names)} BEV embeddings")
        print(f"Failed: {len(failed)} videos")

    else:
        # Original single-camera mode
        # ... existing code ...
```

---

### 10. `pipeline/__init__.py` 수정

**추가할 export:**
```python
from .bev import BEVConfig, BEVGenerator, BEVStitcher, CalibrationLoader
from .embedder.bev_embedder import BEVEmbedder
from .embedder.multicam_loader import MultiCameraVideoInfo, find_multicam_videos

__all__ = [
    # ... existing exports ...

    # BEV
    "BEVConfig",
    "BEVGenerator",
    "BEVStitcher",
    "CalibrationLoader",
    "BEVEmbedder",
    "MultiCameraVideoInfo",
    "find_multicam_videos",
]
```

---

## 의존성 추가

**requirements.txt에 추가:**
```
opencv-python>=4.5.0
PyYAML>=6.0
```

---

## 구현 순서

1. `pipeline/bev/config.py`
2. `pipeline/bev/calibration.py`
3. `pipeline/bev/ipm.py`
4. `pipeline/bev/stitcher.py`
5. `pipeline/bev/generator.py`
6. `pipeline/bev/__init__.py`
7. `pipeline/embedder/multicam_loader.py`
8. `pipeline/embedder/bev_embedder.py`
9. `pipeline/cli.py` 수정
10. `pipeline/__init__.py` 수정
11. `requirements.txt` 업데이트

---

## 사용 예시

```bash
# BEV 모드로 실행
python video_curation_pipeline.py \
    --frame-dir ./trainlake \
    --bev-mode \
    --calibration-dir ./calibrations \
    --output-dir ./output_bev \
    --num-frames 16 \
    --bev-width 800 \
    --bev-height 800 \
    --bev-range 25.0
```

---

## 데이터 흐름 다이어그램

```
trainlake/24.01w/N1234567-240115120000/RAW_DB/
├── FRONT_LONG_CMR/CMR_GT_Frame/     ─┐
├── FRONT_NARROW_CMR/CMR_GT_Frame/   ─┤
├── FRONT_WIDE_CMR/CMR_GT_Frame/     ─┤
├── LEFT_FRONT_CMR/CMR_GT_Frame/     ─┤
├── LEFT_REAR_CMR/CMR_GT_Frame/      ─┤
├── LEFT_WIDE_CMR/CMR_GT_Frame/      ─┼─> find_multicam_videos()
├── RIGHT_FRONT_CMR/CMR_GT_Frame/    ─┤         │
├── RIGHT_REAR_CMR/CMR_GT_Frame/     ─┤         ▼
├── RIGHT_WIDE_CMR/CMR_GT_Frame/     ─┤  MultiCameraVideoInfo
├── REAR_NARROW_CMR/CMR_GT_Frame/    ─┤         │
└── REAR_WIDE_CMR/CMR_GT_Frame/      ─┘         ▼
                                         BEVGenerator.generate_bev_sequence()
calibrations/N1234567/calibration.json ────────┤
                                               ▼
                                    11개 카메라 IPM 변환 + 스티칭
                                               │
                                               ▼
                                      BEV 이미지 (800x800x3)
                                               │
                                               ▼
                                      InternVL Vision Encoder
                                               │
                                               ▼
                                      임베딩 (4096,) 벡터
                                               │
                                               ▼
                              output/N1234567-240115120000.npy
```
