"""
Centralised configuration.
All tuneable values live here — override via environment variables or .env
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Server ────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 5000))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production")

    # ── Camera ────────────────────────────────────────────────────────────
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", 320))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", 240))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", 30))
    USE_PICAMERA: bool = os.getenv("USE_PICAMERA", "true").lower() == "true"
    WEBCAM_INDEX: int = int(os.getenv("WEBCAM_INDEX", 0))

    # ── Detection ─────────────────────────────────────────────────────────
    MODEL_NAME: str = os.getenv("MODEL_NAME", "yolov8n.pt")
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/yolov8n.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", 0.45))
    MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", 20))
    DETECTION_SKIP_FRAMES: int = int(os.getenv("DETECTION_SKIP_FRAMES", 10))
    DEVICE: str = os.getenv("DEVICE", "cpu")

    # ── Streaming ─────────────────────────────────────────────────────────
    STREAM_QUALITY: int = int(os.getenv("STREAM_QUALITY", 50))
    STREAM_MAX_FPS: int = int(os.getenv("STREAM_MAX_FPS", 12))

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/pi-detect.log")
