from abc import ABC, abstractmethod

class CaptioningInterface(ABC):
    """Abstract base class for video captioning interfaces."""

    @abstractmethod
    def caption_video(
        self,
        video_path: str,
        cluster_id: int
    ) -> str:
        """Generate caption for a video."""
        pass

    def generate_cluster_captions(
        self,
        representative_videos: dict[int, str]
    ) -> dict[int, str]:
        """
        Generate captions for all representative videos.

        Args:
            representative_videos: Mapping of cluster_id to video path.

        Returns:
            Mapping of cluster_id to caption string.
        """
        captions = {}
        for cluster_id, video_path in representative_videos.items():
            captions[cluster_id] = self.caption_video(video_path, cluster_id)
        return captions