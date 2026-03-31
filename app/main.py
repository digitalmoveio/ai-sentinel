"""
AI Sentinel - Flask Application

Web interface and API for AI-generated content detection.
"""

import logging
import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from .detectors import ContentAnalyzer

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / "static" / "uploads"
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif", "gif",
    "mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "m4v",
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.secret_key = os.environ.get("SECRET_KEY", "ai-sentinel-dev-key")

    # Ensure upload directory exists
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    # Initialize the content analyzer
    analyzer = ContentAnalyzer()

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    # ------------------------------------------------------------------ #
    #  Web Routes
    # ------------------------------------------------------------------ #

    @app.route("/")
    def index():
        """Render the main page."""
        return render_template("index.html")

    @app.route("/about")
    def about():
        """Render the about page."""
        return render_template("about.html")

    # ------------------------------------------------------------------ #
    #  API Routes
    # ------------------------------------------------------------------ #

    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        """
        Analyze an uploaded file for AI-generated content.

        Accepts multipart/form-data with a 'file' field.

        Returns JSON with detection results.
        """
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not supported. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        # Save with unique name
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = UPLOAD_FOLDER / unique_name

        try:
            file.save(str(filepath))
            logger.info("Analyzing file: %s", filename)

            result = analyzer.analyze(str(filepath))
            result["original_filename"] = filename

            return jsonify(result)

        except Exception as e:
            logger.error("Analysis failed: %s", e)
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

        finally:
            # Clean up uploaded file
            if filepath.exists():
                filepath.unlink()

    @app.route("/api/formats", methods=["GET"])
    def api_formats():
        """Return supported file formats."""
        return jsonify(ContentAnalyzer.supported_formats())

    @app.route("/api/health", methods=["GET"])
    def api_health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "version": "0.1.0"})

    return app


def main():
    """Run the application."""
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
