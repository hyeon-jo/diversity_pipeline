"""Data types and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClusterInfo:
    """Information about a single cluster."""
    cluster_id: int
    size: int
    centroid: NDArray[np.float32]
    density: float
    representative_idx: int
    representative_path: Optional[str] = None
    is_micro_cluster: bool = False
    is_low_density: bool = False


@dataclass
class PipelineResults:
    """Complete results from the curation pipeline."""
    embeddings: NDArray[np.float32]
    cluster_labels: NDArray[np.int32]
    cluster_info: dict[int, ClusterInfo]
    representative_videos: dict[int, str]
    edge_case_indices: list[int]
    void_cluster_ids: list[int]
    captions: dict[int, str] = field(default_factory=dict)
