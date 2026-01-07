from __future__ import annotations
import numpy as np
from .base import CaptioningInterface

class MockCaptioningInterface(CaptioningInterface):
    """
    Mock captioning interface for demonstration.

    In production, this would be replaced with actual VLM API calls
    (e.g., GPT-4V, Claude Vision, or specialized video understanding models).
    """

    # Simulated scenario types for demo
    SCENARIO_TYPES = [
        "Rainy night on highway",
        "Dense urban traffic with pedestrians",
        "Construction zone with lane closures",
        "Parking lot navigation",
        "Highway merge with fast traffic",
        "Foggy morning on rural road",
        "School zone during dismissal",
        "Roundabout navigation",
        "Emergency vehicle approaching",
        "Cyclist in bike lane",
        "Deer crossing warning area",
        "Tunnel entrance with lighting change",
        "Snow-covered residential street",
        "Loading dock area",
        "Multi-lane intersection",
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize mock captioning interface.

        Args:
            seed: Random seed for reproducible captions.
        """
        self.rng = np.random.default_rng(seed)

    def caption_video(
        self,
        video_path: str,
        cluster_id: int
    ) -> str:
        """
        Generate a mock caption for a video.
        """
        # Mock: Select scenario based on cluster_id for consistency
        scenario_idx = cluster_id % len(self.SCENARIO_TYPES)
        scenario = self.SCENARIO_TYPES[scenario_idx]

        return f"Cluster {cluster_id}: {scenario}"