"""
Image AI Detection Engine

Uses multiple analysis techniques to determine if an image is AI-generated:
1. Error Level Analysis (ELA) - Detects compression inconsistencies
2. Frequency Domain Analysis (FFT) - AI images have distinct frequency patterns
3. Statistical Analysis - Pixel distribution and noise patterns
4. Metadata Analysis - EXIF data anomalies common in AI-generated images
5. Texture Analysis - Local Binary Patterns for texture consistency
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ExifTags
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)


def _sanitize_for_json(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class DetectionResult:
    """Result of an AI detection analysis."""

    is_ai_generated: bool
    ai_probability: float  # 0.0 (real) to 1.0 (AI-generated)
    confidence: float  # How confident the system is in the result
    analysis_details: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def real_probability(self) -> float:
        return 1.0 - self.ai_probability

    def to_dict(self) -> dict:
        return {
            "is_ai_generated": bool(self.is_ai_generated),
            "ai_probability": round(float(self.ai_probability) * 100, 2),
            "real_probability": round(float(self.real_probability) * 100, 2),
            "confidence": round(float(self.confidence) * 100, 2),
            "analysis_details": _sanitize_for_json(self.analysis_details),
            "warnings": self.warnings,
        }


class ImageDetector:
    """Multi-method AI image detection engine."""

    # Weights for combining detection methods
    WEIGHTS = {
        "ela": 0.25,
        "frequency": 0.25,
        "statistical": 0.20,
        "metadata": 0.15,
        "texture": 0.15,
    }

    def __init__(self):
        self._initialized = True
        logger.info("ImageDetector initialized with %d analysis methods", len(self.WEIGHTS))

    def analyze(self, image_path: str) -> DetectionResult:
        """
        Analyze an image for AI-generated content.

        Args:
            image_path: Path to the image file.

        Returns:
            DetectionResult with AI probability and analysis details.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to open image: %s", e)
            return DetectionResult(
                is_ai_generated=False,
                ai_probability=0.0,
                confidence=0.0,
                warnings=[f"Could not open image: {str(e)}"],
            )

        img_array = np.array(img)
        scores = {}
        details = {}
        warnings = []

        # --- 1. Error Level Analysis ---
        try:
            ela_score, ela_detail = self._error_level_analysis(img)
            scores["ela"] = ela_score
            details["error_level_analysis"] = ela_detail
        except Exception as e:
            warnings.append(f"ELA analysis failed: {str(e)}")
            logger.warning("ELA failed: %s", e)

        # --- 2. Frequency Domain Analysis ---
        try:
            freq_score, freq_detail = self._frequency_analysis(img_array)
            scores["frequency"] = freq_score
            details["frequency_analysis"] = freq_detail
        except Exception as e:
            warnings.append(f"Frequency analysis failed: {str(e)}")
            logger.warning("Frequency analysis failed: %s", e)

        # --- 3. Statistical Analysis ---
        try:
            stat_score, stat_detail = self._statistical_analysis(img_array)
            scores["statistical"] = stat_score
            details["statistical_analysis"] = stat_detail
        except Exception as e:
            warnings.append(f"Statistical analysis failed: {str(e)}")
            logger.warning("Statistical analysis failed: %s", e)

        # --- 4. Metadata Analysis ---
        try:
            meta_score, meta_detail = self._metadata_analysis(image_path, img)
            scores["metadata"] = meta_score
            details["metadata_analysis"] = meta_detail
        except Exception as e:
            warnings.append(f"Metadata analysis failed: {str(e)}")
            logger.warning("Metadata analysis failed: %s", e)

        # --- 5. Texture Analysis ---
        try:
            tex_score, tex_detail = self._texture_analysis(img_array)
            scores["texture"] = tex_score
            details["texture_analysis"] = tex_detail
        except Exception as e:
            warnings.append(f"Texture analysis failed: {str(e)}")
            logger.warning("Texture analysis failed: %s", e)

        # --- Combine scores ---
        if not scores:
            return DetectionResult(
                is_ai_generated=False,
                ai_probability=0.0,
                confidence=0.0,
                warnings=["All analysis methods failed"],
            )

        ai_probability = self._combine_scores(scores)
        confidence = len(scores) / len(self.WEIGHTS)

        return DetectionResult(
            is_ai_generated=ai_probability > 0.5,
            ai_probability=ai_probability,
            confidence=confidence,
            analysis_details=details,
            warnings=warnings,
        )

    def _combine_scores(self, scores: dict) -> float:
        """Weighted combination of individual detection scores."""
        total_weight = sum(self.WEIGHTS[k] for k in scores)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        return np.clip(weighted_sum / total_weight, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Detection Method 1: Error Level Analysis (ELA)
    # ------------------------------------------------------------------ #
    def _error_level_analysis(self, img: Image.Image, quality: int = 90) -> tuple:
        """
        Resave the image at a known JPEG quality and compare.
        AI-generated images often show uniform error levels, while
        real photos have varied compression artifacts.
        """
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        original_arr = np.array(img, dtype=np.float64)
        resaved_arr = np.array(resaved, dtype=np.float64)

        # Error level = absolute difference scaled
        ela = np.abs(original_arr - resaved_arr)
        ela_scaled = (ela * 255.0 / ela.max()) if ela.max() > 0 else ela

        # AI images tend to have more uniform ELA
        mean_ela = np.mean(ela_scaled)
        std_ela = np.std(ela_scaled)
        uniformity = 1.0 - min(std_ela / (mean_ela + 1e-10), 1.0)

        # High uniformity → more likely AI
        # Also check if mean error is suspiciously low (over-smooth)
        smoothness_score = 1.0 - np.clip(mean_ela / 128.0, 0.0, 1.0)

        score = 0.6 * uniformity + 0.4 * smoothness_score

        detail = {
            "mean_error_level": round(float(mean_ela), 2),
            "error_std_dev": round(float(std_ela), 2),
            "uniformity": round(float(uniformity), 4),
            "smoothness": round(float(smoothness_score), 4),
            "score": round(float(score), 4),
            "interpretation": "High uniformity in error levels suggests AI generation"
            if score > 0.5
            else "Error level patterns are consistent with a real photograph",
        }

        return float(score), detail

    # ------------------------------------------------------------------ #
    #  Detection Method 2: Frequency Domain Analysis (FFT)
    # ------------------------------------------------------------------ #
    def _frequency_analysis(self, img_array: np.ndarray) -> tuple:
        """
        Analyze the frequency spectrum using FFT.
        AI-generated images often lack certain high-frequency details
        and show periodic patterns in the frequency domain.
        """
        gray = np.mean(img_array, axis=2)

        # 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log1p(np.abs(f_shift))

        # Analyze radial frequency distribution
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        radial_profile = []
        for r in range(1, max_radius, max(1, max_radius // 50)):
            y, x = np.ogrid[-cy : h - cy, -cx : w - cx]
            mask = (x * x + y * y >= (r - 1) ** 2) & (x * x + y * y < r**2)
            if mask.any():
                radial_profile.append(float(np.mean(magnitude[mask])))

        if len(radial_profile) < 3:
            return 0.5, {"score": 0.5, "interpretation": "Image too small for frequency analysis"}

        radial_profile = np.array(radial_profile)

        # AI images: faster high-frequency rolloff
        mid = len(radial_profile) // 2
        low_freq_energy = np.mean(radial_profile[:mid])
        high_freq_energy = np.mean(radial_profile[mid:])

        if low_freq_energy > 0:
            rolloff_ratio = high_freq_energy / low_freq_energy
        else:
            rolloff_ratio = 1.0

        # AI images typically have rolloff_ratio < 0.3
        # Real images typically have rolloff_ratio > 0.4
        score = np.clip(1.0 - (rolloff_ratio / 0.6), 0.0, 1.0)

        # Check for periodic artifacts (grid-like patterns from GANs)
        profile_fft = np.abs(np.fft.fft(radial_profile - np.mean(radial_profile)))
        periodicity = float(np.max(profile_fft[1:]) / (np.mean(profile_fft[1:]) + 1e-10))
        periodicity_score = np.clip(periodicity / 10.0, 0.0, 1.0)

        combined_score = 0.7 * score + 0.3 * periodicity_score

        detail = {
            "low_freq_energy": round(float(low_freq_energy), 2),
            "high_freq_energy": round(float(high_freq_energy), 2),
            "rolloff_ratio": round(float(rolloff_ratio), 4),
            "periodicity": round(float(periodicity), 4),
            "score": round(float(combined_score), 4),
            "interpretation": "Frequency spectrum shows AI-typical patterns"
            if combined_score > 0.5
            else "Frequency distribution is consistent with natural photography",
        }

        return float(combined_score), detail

    # ------------------------------------------------------------------ #
    #  Detection Method 3: Statistical Analysis
    # ------------------------------------------------------------------ #
    def _statistical_analysis(self, img_array: np.ndarray) -> tuple:
        """
        Analyze pixel statistics. AI-generated images often have
        different noise characteristics and pixel distributions.
        """
        pixels = img_array.astype(np.float64)

        # Per-channel statistics
        channel_stats = {}
        for i, name in enumerate(["red", "green", "blue"]):
            ch = pixels[:, :, i]
            channel_stats[name] = {
                "mean": round(float(np.mean(ch)), 2),
                "std": round(float(np.std(ch)), 2),
                "skewness": round(float(self._skewness(ch)), 4),
                "kurtosis": round(float(self._kurtosis(ch)), 4),
            }

        # Noise estimation using Laplacian
        gray = np.mean(pixels, axis=2)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        from scipy.signal import convolve2d

        noise_map = convolve2d(gray, laplacian, mode="valid")
        noise_level = np.std(noise_map)

        # AI images often have unnaturally low noise or very uniform noise
        # Natural photos: noise_level typically 5-30
        # AI images: often < 5 or very uniform
        noise_score = 1.0 - np.clip(noise_level / 25.0, 0.0, 1.0)

        # Check histogram smoothness (AI images have smoother histograms)
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
        hist_smoothness = 1.0 - np.std(np.diff(hist.astype(float))) / (np.mean(hist) + 1e-10)
        hist_smoothness = np.clip(hist_smoothness, 0.0, 1.0)

        # Pixel value uniqueness ratio
        unique_ratio = len(np.unique(img_array.reshape(-1, 3), axis=0)) / (
            img_array.shape[0] * img_array.shape[1]
        )
        # AI images often have fewer unique colors relative to size
        uniqueness_score = 1.0 - np.clip(unique_ratio / 0.8, 0.0, 1.0)

        score = 0.4 * noise_score + 0.3 * hist_smoothness + 0.3 * uniqueness_score

        detail = {
            "channel_statistics": channel_stats,
            "noise_level": round(float(noise_level), 2),
            "histogram_smoothness": round(float(hist_smoothness), 4),
            "color_uniqueness_ratio": round(float(unique_ratio), 4),
            "score": round(float(score), 4),
            "interpretation": "Statistical patterns suggest AI generation"
            if score > 0.5
            else "Pixel statistics are consistent with natural photography",
        }

        return float(score), detail

    # ------------------------------------------------------------------ #
    #  Detection Method 4: Metadata Analysis
    # ------------------------------------------------------------------ #
    def _metadata_analysis(self, image_path: str, img: Image.Image) -> tuple:
        """
        Analyze EXIF and metadata. AI-generated images typically
        lack camera metadata or have suspicious metadata patterns.
        """
        indicators = []
        score_components = []

        # Check EXIF data
        exif_data = {}
        try:
            raw_exif = img._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        continue
                    exif_data[str(tag)] = str(value)
        except (AttributeError, Exception):
            pass

        # No EXIF = suspicious (but not conclusive)
        if not exif_data:
            indicators.append("No EXIF metadata found")
            score_components.append(0.6)
        else:
            # Check for camera-related fields
            camera_fields = ["Make", "Model", "LensModel", "FocalLength", "ExposureTime", "FNumber"]
            found_camera = sum(1 for f in camera_fields if f in exif_data)

            if found_camera == 0:
                indicators.append("No camera information in EXIF")
                score_components.append(0.7)
            elif found_camera < 3:
                indicators.append(f"Partial camera info ({found_camera}/{len(camera_fields)} fields)")
                score_components.append(0.4)
            else:
                indicators.append(f"Full camera metadata present ({found_camera} fields)")
                score_components.append(0.1)

            # Check for AI tool signatures
            ai_tools = [
                "stable diffusion", "midjourney", "dall-e", "dalle",
                "comfyui", "automatic1111", "novelai", "adobe firefly",
                "ideogram", "flux", "leonardo", "playground",
            ]
            software = exif_data.get("Software", "").lower()
            description = exif_data.get("ImageDescription", "").lower()
            combined_text = software + " " + description

            for tool in ai_tools:
                if tool in combined_text:
                    indicators.append(f"AI tool signature detected: {tool}")
                    score_components.append(1.0)
                    break

        # Check file size vs resolution ratio
        try:
            file_size = Path(image_path).stat().st_size
            pixel_count = img.size[0] * img.size[1]
            bytes_per_pixel = file_size / pixel_count if pixel_count > 0 else 0

            # AI-generated PNGs tend to be larger per pixel
            # AI-generated JPEGs can be unusually uniform in compression
            if bytes_per_pixel > 4.0:
                indicators.append("Unusually high bytes-per-pixel ratio")
                score_components.append(0.55)
            elif bytes_per_pixel < 0.3:
                indicators.append("Very low bytes-per-pixel (heavy compression)")
                score_components.append(0.3)
        except Exception:
            pass

        score = float(np.mean(score_components)) if score_components else 0.5

        detail = {
            "has_exif": bool(exif_data),
            "exif_fields_count": len(exif_data),
            "indicators": indicators,
            "score": round(score, 4),
            "interpretation": "Metadata patterns suggest AI-generated content"
            if score > 0.5
            else "Metadata is consistent with a real photograph",
        }

        return score, detail

    # ------------------------------------------------------------------ #
    #  Detection Method 5: Texture Analysis (LBP-based)
    # ------------------------------------------------------------------ #
    def _texture_analysis(self, img_array: np.ndarray) -> tuple:
        """
        Local Binary Pattern analysis for texture consistency.
        AI-generated images often have unnaturally consistent textures.
        """
        gray = np.mean(img_array, axis=2).astype(np.uint8)

        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i - 1, j - 1] >= center) << 7
                code |= (gray[i - 1, j] >= center) << 6
                code |= (gray[i - 1, j + 1] >= center) << 5
                code |= (gray[i, j + 1] >= center) << 4
                code |= (gray[i + 1, j + 1] >= center) << 3
                code |= (gray[i + 1, j] >= center) << 2
                code |= (gray[i + 1, j - 1] >= center) << 1
                code |= (gray[i, j - 1] >= center) << 0
                lbp[i - 1, j - 1] = code

        # For large images, subsample to speed up
        if lbp.size > 500 * 500:
            step = max(1, int(np.sqrt(lbp.size / (500 * 500))))
            lbp = lbp[::step, ::step]

        # Analyze LBP histogram
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 255), density=True)

        # Entropy of LBP histogram
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        # AI images often have lower texture entropy (more uniform textures)
        # Natural images: entropy typically 5-7
        # AI images: often 3-5
        max_entropy = np.log2(256)  # ~8
        normalized_entropy = entropy / max_entropy
        texture_score = 1.0 - np.clip(normalized_entropy / 0.85, 0.0, 1.0)

        # Regional texture consistency
        block_size = min(lbp.shape[0], lbp.shape[1]) // 4
        if block_size > 10:
            block_entropies = []
            for bi in range(0, lbp.shape[0] - block_size, block_size):
                for bj in range(0, lbp.shape[1] - block_size, block_size):
                    block = lbp[bi : bi + block_size, bj : bj + block_size]
                    bh, _ = np.histogram(block.flatten(), bins=256, range=(0, 255), density=True)
                    bh_nz = bh[bh > 0]
                    block_entropies.append(-np.sum(bh_nz * np.log2(bh_nz)))

            if block_entropies:
                consistency = 1.0 - (np.std(block_entropies) / (np.mean(block_entropies) + 1e-10))
                consistency = np.clip(consistency, 0.0, 1.0)
            else:
                consistency = 0.5
        else:
            consistency = 0.5

        score = 0.5 * texture_score + 0.5 * consistency

        detail = {
            "lbp_entropy": round(float(entropy), 4),
            "normalized_entropy": round(float(normalized_entropy), 4),
            "texture_consistency": round(float(consistency), 4),
            "score": round(float(score), 4),
            "interpretation": "Texture patterns suggest AI generation"
            if score > 0.5
            else "Texture patterns are consistent with natural photography",
        }

        return float(score), detail

    # ------------------------------------------------------------------ #
    #  Utility Methods
    # ------------------------------------------------------------------ #
    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return float(np.mean(((arr - mean) / std) ** 4) - 3.0)
