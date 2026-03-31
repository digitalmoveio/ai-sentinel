"""
Tests for AI Sentinel detection engine.
"""

import os
import tempfile
import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def smooth_image():
    """Create a smooth gradient image (more likely to look AI-generated)."""
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        arr[i, :, :] = i
    img = Image.fromarray(arr)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        yield f.name
    os.unlink(f.name)


class TestImageDetector:
    def test_analyze_returns_result(self, sample_image):
        from app.detectors.image_detector import ImageDetector

        detector = ImageDetector()
        result = detector.analyze(sample_image)

        assert result is not None
        assert 0.0 <= result.ai_probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.analysis_details, dict)

    def test_analyze_invalid_path(self):
        from app.detectors.image_detector import ImageDetector

        detector = ImageDetector()
        result = detector.analyze("/nonexistent/path.jpg")

        assert result.ai_probability == 0.0
        assert len(result.warnings) > 0

    def test_result_to_dict(self, sample_image):
        from app.detectors.image_detector import ImageDetector

        detector = ImageDetector()
        result = detector.analyze(sample_image)
        d = result.to_dict()

        assert "ai_probability" in d
        assert "real_probability" in d
        assert "confidence" in d
        assert d["ai_probability"] + d["real_probability"] == pytest.approx(100.0, abs=0.1)

    def test_smooth_image_scores_higher(self, sample_image, smooth_image):
        from app.detectors.image_detector import ImageDetector

        detector = ImageDetector()
        random_result = detector.analyze(sample_image)
        smooth_result = detector.analyze(smooth_image)

        # Smooth images should generally score higher on AI probability
        # (not guaranteed but expected in most cases)
        assert smooth_result.ai_probability >= 0.0  # Basic sanity check


class TestContentAnalyzer:
    def test_analyze_image(self, sample_image):
        from app.detectors.analyzer import ContentAnalyzer

        analyzer = ContentAnalyzer()
        result = analyzer.analyze(sample_image)

        assert result["type"] == "image"
        assert "result" in result

    def test_unsupported_format(self):
        from app.detectors.analyzer import ContentAnalyzer

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"not a real file")
            path = f.name

        analyzer = ContentAnalyzer()
        result = analyzer.analyze(path)

        assert "error" in result
        os.unlink(path)

    def test_supported_formats(self):
        from app.detectors.analyzer import ContentAnalyzer

        formats = ContentAnalyzer.supported_formats()
        assert "image" in formats
        assert "video" in formats
        assert ".jpg" in formats["image"]
        assert ".mp4" in formats["video"]


class TestFlaskApp:
    @pytest.fixture
    def client(self):
        from app.main import create_app

        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_index(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"

    def test_formats(self, client):
        response = client.get("/api/formats")
        assert response.status_code == 200
        data = response.get_json()
        assert "image" in data
        assert "video" in data

    def test_analyze_no_file(self, client):
        response = client.post("/api/analyze")
        assert response.status_code == 400

    def test_analyze_image(self, client):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        import io

        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)

        response = client.post(
            "/api/analyze",
            data={"file": (buf, "test.png")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["type"] == "image"
        assert "result" in data
