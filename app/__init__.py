"""
App factory — creates and wires up the Flask application.
"""

from flask import Flask
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_app(detection_enabled: bool = True) -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(Config)
    app.config["DETECTION_ENABLED"] = detection_enabled

    from app.streaming.camera import CameraStream
    from app.detection.detector import ObjectDetector

    camera = CameraStream()
    detector = ObjectDetector() if detection_enabled else None

    app.camera = camera
    app.detector = detector

    from app.streaming.stream import stream_bp
    from app.routes import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(stream_bp)

    import atexit

    def _shutdown():
        logger.info("Shutting down camera...")
        camera.stop()
        if detector:
            detector.stop()

    atexit.register(_shutdown)
    logger.info("Flask app created successfully.")
    return app
