"""
Camera abstraction layer.
Supports PiCamera2 (Raspberry Pi) and OpenCV webcam (dev machine).
Runs in a background daemon thread.
"""

import time
import threading
import numpy as np
import cv2
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CameraStream:
    """Thread-safe camera that keeps the latest frame in memory."""

    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.actual_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.time()
        self._start()

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Camera stopped.")

    def _start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera-thread"
        )
        self._thread.start()
        logger.info("Camera thread started (picamera2=%s)", Config.USE_PICAMERA)

    def _capture_loop(self):
        if Config.USE_PICAMERA:
            self._run_picamera()
        else:
            self._run_opencv()

    def _run_picamera(self):
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_video_configuration(
                main={
                    "size": (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT),
                    "format": "RGB888",
                },
                controls={"FrameRate": Config.CAMERA_FPS},
            )
            cam.configure(config)
            cam.start()
            logger.info(
                "PiCamera2 started — %dx%d @ %dfps",
                Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT, Config.CAMERA_FPS,
            )
            while self._running:
                frame_rgb = cam.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self._update_frame(frame_bgr)
            cam.stop()
        except Exception as exc:
            logger.error("PiCamera2 error: %s — falling back to OpenCV", exc)
            self._run_opencv()

    def _run_opencv(self):
        cap = cv2.VideoCapture(Config.WEBCAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, Config.CAMERA_FPS)
        if not cap.isOpened():
            logger.error("Cannot open camera index %d", Config.WEBCAM_INDEX)
            return
        logger.info("OpenCV webcam started — index %d", Config.WEBCAM_INDEX)
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            self._update_frame(frame)
        cap.release()

    def _update_frame(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self.actual_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()
