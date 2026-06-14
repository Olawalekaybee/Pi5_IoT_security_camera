"""
Main routes — dashboard UI and API endpoints.
"""

from flask import Blueprint, render_template, jsonify, current_app

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@main_bp.route("/api/stats")
def stats():
    detector = current_app.detector
    camera = current_app.camera
    return jsonify({
        "fps": round(camera.actual_fps, 1),
        "detections": detector.last_detection_count if detector else 0,
        "inference_ms": round(detector.last_inference_ms, 1) if detector else 0,
        "detection_enabled": current_app.config["DETECTION_ENABLED"],
        "objects": detector.last_labels if detector else [],
    })


@main_bp.route("/api/config")
def get_config():
    from app.config import Config
    return jsonify({
        "resolution": f"{Config.CAMERA_WIDTH}x{Config.CAMERA_HEIGHT}",
        "model": Config.MODEL_NAME,
        "confidence": Config.CONFIDENCE_THRESHOLD,
        "skip_frames": Config.DETECTION_SKIP_FRAMES,
        "stream_fps": Config.STREAM_MAX_FPS,
    })


@main_bp.route("/api/health")
def health():
    return jsonify({"status": "ok"}), 200
