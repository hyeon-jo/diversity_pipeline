"""Mock captioning interface for demonstration purposes."""

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

        In production, replace this with:

        ```python
        # Example GPT-4V API call (pseudocode)
        from openai import OpenAI

        client = OpenAI()

        # Extract key frames from video
        frames = extract_key_frames(video_path, n_frames=4)

        # Encode frames as base64
        images = [encode_image(frame) for frame in frames]

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this driving scenario..."},
                    *[{"type": "image_url", "image_url": img} for img in images]
                ]
            }]
        )
        return response.choices[0].message.content
        ```

        Args:
            video_path: Path to the video file.
            cluster_id: ID of the cluster this video represents.

        Returns:
            Caption string describing the scenario.
        """
        # Mock: Select scenario based on cluster_id for consistency
        scenario_idx = cluster_id % len(self.SCENARIO_TYPES)
        scenario = self.SCENARIO_TYPES[scenario_idx]

        return f"Cluster {cluster_id}: {scenario}"
