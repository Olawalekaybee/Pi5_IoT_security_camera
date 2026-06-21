"""
Edge AI Security Monitor
Real-time person detection + re-identification on Raspberry Pi 5 + Hailo-8L HAT
"""

import argparse
import signal
import sys
import threading
import logging
from pathlib import Path

from src.detection.pipeline import DetectionPipeline
from src.reid.identifier import PersonIdentifier
from src.alerts.telegram_bot import TelegramAlert
from src.dashboard.server import DashboardServer
from src.utils.database import Database
from src.utils.logger import setup_logger
from src.utils.liveness import LivenessTracker
from config.settings import Settings


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Edge AI Security Monitor")
    parser.add_argument("--config", type=str, default="config/settings.yaml",
                        help="Path to config file")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index or RTSP URL")
    parser.add_argument("--no-alerts", action="store_true",
                        help="Disable Telegram alerts (useful for testing)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable web dashboard")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode — prints FPS/latency stats")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(args.log_level)

    logger.info("=== Edge AI Security Monitor starting ===")

    # Load config
    settings = Settings.from_yaml(args.config)

    # Init shared database
    db = Database(settings.database.path)
    db.init_tables()

    # Init core components
    pipeline = DetectionPipeline(
        model_path=settings.models.detection_hef,
        camera_index=args.camera,
        zones=settings.zones,
        settings=settings,
    )

    identifier = PersonIdentifier(
        model_path=settings.models.reid_hef,
        known_faces_dir=settings.known_faces_dir,
        threshold=settings.reid.similarity_threshold,
    )

    alert = TelegramAlert(
        token=settings.telegram.bot_token,
        chat_id=settings.telegram.chat_id,
        enabled=not args.no_alerts,
    )

    # Motion-based liveness heuristic — see src/utils/liveness.py for
    # an honest explanation of what this can and can't actually catch.
    liveness = LivenessTracker()

    # Wire components together
    pipeline.on_detection = lambda event: handle_detection(
        event, identifier, alert, db, settings, liveness
    )

    # Start dashboard in background thread
    dashboard = None
    if not args.no_dashboard:
        dashboard = DashboardServer(db=db, settings=settings, pipeline=pipeline, identifier=identifier)
        dashboard_thread = threading.Thread(
            target=dashboard.run, daemon=True, name="dashboard"
        )
        dashboard_thread.start()
        logger.info(f"Dashboard running at http://0.0.0.0:{settings.dashboard.port}")

    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info("Shutting down...")
        pipeline.stop()
        identifier.close()
        db.close()
        from src.detection.hailo_engine import release_shared_vdevice
        release_shared_vdevice()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start main pipeline (blocking)
    if args.benchmark:
        pipeline.run_benchmark()
    else:
        pipeline.run()


def handle_detection(event, identifier, alert, db, settings, liveness):
    """
    Called on every detection event from the pipeline.
    Runs Re-ID, liveness heuristic, decides alert, writes to DB.
    """
    person_id, confidence = identifier.identify(event.crop)
    event.person_id = person_id
    event.reid_confidence = confidence

    is_unknown = person_id is None or confidence < settings.reid.alert_threshold
    in_restricted = event.zone in settings.zones.restricted

    # Liveness only matters for someone who'd otherwise be trusted —
    # an already-unknown person triggers an alert regardless. Track
    # motion under the *claimed* identity so a photo of a known person
    # held up to the camera still gets flagged, even though Re-ID
    # correctly matches the face in the photo.
    spoof_suspected = False
    if person_id is not None:
        liveness.update(person_id, event.bbox)
        is_static = liveness.is_suspiciously_static(person_id)
        event.liveness_static = is_static
        if is_static is True:
            spoof_suspected = True

    db.insert_event(event)

    if is_unknown and in_restricted:
        logger.warning(
            f"Unknown person in restricted zone '{event.zone}' "
            f"(Re-ID conf: {confidence:.2f})"
        )
        alert.send(event)
    elif spoof_suspected and in_restricted:
        logger.warning(
            f"Possible spoof: '{person_id}' matched in zone '{event.zone}' "
            f"but motion looks static (likely a photo/screen, not a live person) — "
            f"alerting despite the Re-ID match"
        )
        # Keep person_id intact for display (so the dashboard/video feed
        # can show "olawale (SPOOF?)" rather than losing the identity
        # info), but alert.send() doesn't care about person_id — it
        # alerts based on the event being passed to it regardless.
        alert.send(event)


if __name__ == "__main__":
    main()