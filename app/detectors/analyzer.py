"""
Content Analyzer - Unified interface for analyzing images and videos.
"""

import logging
import mimetypes
from pathlib import Path

from .image_detector import ImageDetector, DetectionResult
from .video_detector import VideoDetector, VideoDetectionResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


class ContentAnalyzer:
    """Unified content analyzer that routes to the appropriate detector."""

    def __init__(self):
        self.image_detector = ImageDetector()
        self.video_detector = VideoDetector(image_detector=self.image_detector)
        logger.info("ContentAnalyzer initialized")

    def analyze(self, file_path: str) -> dict:
        """
        Analyze a file and return detection results.

        Args:
            file_path: Path to the image or video file.

        Returns:
            Dict with detection results.
        """
        path = Path(file_path)

        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        ext = path.suffix.lower()
        media_type = self._detect_media_type(file_path, ext)

        if media_type == "image":
            result = self.image_detector.analyze(file_path)
            return {
                "file": path.name,
                "type": "image",
                "result": result.to_dict(),
            }
        elif media_type == "video":
            result = self.video_detector.analyze(file_path)
            return {
                "file": path.name,
                "type": "video",
                "result": result.to_dict(),
            }
        else:
            return {"error": f"Unsupported file type: {ext}"}

    def _detect_media_type(self, file_path: str, ext: str) -> str:
        """Detect whether the file is an image or video."""
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in VIDEO_EXTENSIONS:
            return "video"

        # Fallback: try mimetypes
        mime, _ = mimetypes.guess_type(file_path)
        if mime:
            if mime.startswith("image/"):
                return "image"
            if mime.startswith("video/"):
                return "video"

        return "unknown"

    @staticmethod
    def supported_formats() -> dict:
        """Return supported file formats."""
        return {
            "image": sorted(IMAGE_EXTENSIONS),
            "video": sorted(VIDEO_EXTENSIONS),
        }
