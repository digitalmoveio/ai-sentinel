"""
Video AI Detection Engine

Analyzes videos by extracting key frames and running image detection,
plus video-specific analyses:
1. Frame-by-frame AI detection (sampling key frames)
2. Temporal consistency analysis (AI videos have flickering/inconsistency)
3. Motion analysis (optical flow patterns differ in AI videos)
4. Audio-visual sync analysis placeholder
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoDetectionResult:
    """Result of video AI detection analysis."""

    is_ai_generated: bool
    ai_probability: float
    confidence: float
    frames_analyzed: int
    total_frames: int
    duration_seconds: float
    frame_results: list = field(default_factory=list)
    temporal_analysis: dict = field(default_factory=dict)
    motion_analysis: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def real_probability(self) -> float:
        return 1.0 - self.ai_probability

    def to_dict(self) -> dict:
        return {
            "is_ai_generated": self.is_ai_generated,
            "ai_probability": round(self.ai_probability * 100, 2),
            "real_probability": round(self.real_probability * 100, 2),
            "confidence": round(self.confidence * 100, 2),
            "frames_analyzed": self.frames_analyzed,
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "temporal_analysis": self.temporal_analysis,
            "motion_analysis": self.motion_analysis,
            "warnings": self.warnings,
        }


class VideoDetector:
    """Video AI detection engine using frame sampling and temporal analysis."""

    MAX_FRAMES_TO_SAMPLE = 20  # Max frames to analyze
    MIN_FRAMES_TO_SAMPLE = 5

    def __init__(self, image_detector=None):
        from .image_detector import ImageDetector

        self.image_detector = image_detector or ImageDetector()
        logger.info("VideoDetector initialized")

    def analyze(self, video_path: str, sample_frames: Optional[int] = None) -> VideoDetectionResult:
        """
        Analyze a video for AI-generated content.

        Args:
            video_path: Path to the video file.
            sample_frames: Number of frames to sample (auto-determined if None).

        Returns:
            VideoDetectionResult with analysis details.
        """
        try:
            import cv2
        except ImportError:
            return VideoDetectionResult(
                is_ai_generated=False,
                ai_probability=0.0,
                confidence=0.0,
                frames_analyzed=0,
                total_frames=0,
                duration_seconds=0.0,
                warnings=["OpenCV not installed. Run: pip install opencv-python-headless"],
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoDetectionResult(
                is_ai_generated=False,
                ai_probability=0.0,
                confidence=0.0,
                frames_analyzed=0,
                total_frames=0,
                duration_seconds=0.0,
                warnings=[f"Could not open video: {video_path}"],
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0.0

        # Determine number of frames to sample
        if sample_frames is None:
            sample_frames = min(
                self.MAX_FRAMES_TO_SAMPLE,
                max(self.MIN_FRAMES_TO_SAMPLE, total_frames // 30),
            )

        # Calculate frame indices to sample (evenly spaced)
        if total_frames <= sample_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int).tolist()

        # --- Extract and analyze frames ---
        frame_results = []
        frame_arrays = []
        warnings = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_arrays.append(frame_rgb)

            # Save frame temporarily and analyze
            try:
                from PIL import Image

                pil_img = Image.fromarray(frame_rgb)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pil_img.save(tmp.name)
                    result = self.image_detector.analyze(tmp.name)
                    frame_results.append(
                        {
                            "frame_index": idx,
                            "timestamp": round(idx / fps, 2),
                            "ai_probability": round(result.ai_probability * 100, 2),
                        }
                    )
                    Path(tmp.name).unlink(missing_ok=True)
            except Exception as e:
                warnings.append(f"Frame {idx} analysis failed: {str(e)}")

        cap.release()

        if not frame_results:
            return VideoDetectionResult(
                is_ai_generated=False,
                ai_probability=0.0,
                confidence=0.0,
                frames_analyzed=0,
                total_frames=total_frames,
                duration_seconds=duration,
                warnings=["No frames could be analyzed"],
            )

        # --- Frame-level AI probability ---
        frame_probs = [r["ai_probability"] / 100.0 for r in frame_results]
        avg_frame_prob = float(np.mean(frame_probs))

        # --- Temporal Consistency Analysis ---
        temporal = self._temporal_consistency(frame_arrays)

        # --- Motion Analysis ---
        motion = self._motion_analysis(frame_arrays)

        # --- Combine all signals ---
        combined_prob = (
            0.50 * avg_frame_prob
            + 0.30 * temporal.get("score", 0.5)
            + 0.20 * motion.get("score", 0.5)
        )
        combined_prob = float(np.clip(combined_prob, 0.0, 1.0))
        confidence = len(frame_results) / sample_frames

        return VideoDetectionResult(
            is_ai_generated=combined_prob > 0.5,
            ai_probability=combined_prob,
            confidence=confidence,
            frames_analyzed=len(frame_results),
            total_frames=total_frames,
            duration_seconds=duration,
            frame_results=frame_results,
            temporal_analysis=temporal,
            motion_analysis=motion,
            warnings=warnings,
        )

    def _temporal_consistency(self, frames: list) -> dict:
        """
        Analyze temporal consistency between frames.
        AI-generated videos often have flickering or inconsistent details.
        """
        if len(frames) < 2:
            return {"score": 0.5, "interpretation": "Not enough frames for temporal analysis"}

        diffs = []
        for i in range(1, len(frames)):
            prev = frames[i - 1].astype(np.float64)
            curr = frames[i].astype(np.float64)

            # Resize if different sizes
            if prev.shape != curr.shape:
                min_h = min(prev.shape[0], curr.shape[0])
                min_w = min(prev.shape[1], curr.shape[1])
                prev = prev[:min_h, :min_w]
                curr = curr[:min_h, :min_w]

            diff = np.mean(np.abs(prev - curr))
            diffs.append(diff)

        diffs = np.array(diffs)
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))

        # AI videos: either too consistent (low std) or flickering (high std relative to mean)
        if mean_diff > 0:
            variation_coefficient = std_diff / mean_diff
        else:
            variation_coefficient = 0.0

        # Detect sudden jumps (flickering)
        if len(diffs) > 2:
            jumps = np.abs(np.diff(diffs))
            flicker_score = float(np.mean(jumps > 2 * std_diff)) if std_diff > 0 else 0.0
        else:
            flicker_score = 0.0

        # Score: high flickering or unnatural consistency → AI
        consistency_anomaly = abs(variation_coefficient - 0.5)  # 0.5 is "normal"
        score = 0.5 * np.clip(consistency_anomaly * 2, 0, 1) + 0.5 * flicker_score

        return {
            "mean_frame_difference": round(mean_diff, 2),
            "frame_diff_std": round(std_diff, 2),
            "variation_coefficient": round(variation_coefficient, 4),
            "flicker_score": round(flicker_score, 4),
            "score": round(float(score), 4),
            "interpretation": "Temporal patterns suggest AI generation"
            if score > 0.5
            else "Frame-to-frame consistency appears natural",
        }

    def _motion_analysis(self, frames: list) -> dict:
        """
        Analyze motion patterns between frames.
        AI-generated videos often have unnatural motion patterns.
        """
        if len(frames) < 2:
            return {"score": 0.5, "interpretation": "Not enough frames for motion analysis"}

        try:
            import cv2
        except ImportError:
            return {"score": 0.5, "interpretation": "OpenCV required for motion analysis"}

        flow_magnitudes = []
        flow_angles = []

        for i in range(1, min(len(frames), 10)):  # Limit for performance
            prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

            # Resize if needed
            if prev_gray.shape != curr_gray.shape:
                h = min(prev_gray.shape[0], curr_gray.shape[0])
                w = min(prev_gray.shape[1], curr_gray.shape[1])
                prev_gray = prev_gray[:h, :w]
                curr_gray = curr_gray[:h, :w]

            # Resize for speed
            scale = min(1.0, 320.0 / max(prev_gray.shape))
            if scale < 1.0:
                new_size = (int(prev_gray.shape[1] * scale), int(prev_gray.shape[0] * scale))
                prev_gray = cv2.resize(prev_gray, new_size)
                curr_gray = cv2.resize(curr_gray, new_size)

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(float(np.mean(mag)))
            flow_angles.append(float(np.std(ang)))

        if not flow_magnitudes:
            return {"score": 0.5, "interpretation": "Could not compute optical flow"}

        mean_magnitude = float(np.mean(flow_magnitudes))
        magnitude_std = float(np.std(flow_magnitudes))
        mean_angle_spread = float(np.mean(flow_angles))

        # AI videos often have either too uniform or too erratic motion
        motion_uniformity = 1.0 - np.clip(magnitude_std / (mean_magnitude + 1e-10), 0, 1)

        # Very uniform motion is suspicious
        score = np.clip(motion_uniformity * 0.7, 0, 1)

        return {
            "mean_flow_magnitude": round(mean_magnitude, 4),
            "flow_magnitude_std": round(magnitude_std, 4),
            "mean_angle_spread": round(mean_angle_spread, 4),
            "motion_uniformity": round(float(motion_uniformity), 4),
            "score": round(float(score), 4),
            "interpretation": "Motion patterns suggest AI generation"
            if score > 0.5
            else "Motion patterns appear natural",
        }
