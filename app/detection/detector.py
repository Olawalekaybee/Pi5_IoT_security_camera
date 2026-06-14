"""
Async Object Detector — runs YOLOv8n inference in a background thread
so it NEVER blocks the MJPEG stream. Stream runs at full FPS while
detection updates boxes in the background.
"""

import time
import threading
import numpy as np
import cv2
from pathlib import Path
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def _class_color(class_id: int) -> tuple:
    return _PALETTE[class_id % len(_PALETTE)]


class ObjectDetector:
    """Thread-safe async YOLOv8n detector."""

    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()
        self._pending_frame = None
        self._frame_lock = threading.Lock()
        self._boxes = []
        self._boxes_lock = threading.Lock()

        # Public stats (read by /api/stats)
        self.last_detection_count: int = 0
        self.last_inference_ms: float = 0.0
        self.last_labels: list = []

        self._running = True
        self._load_model()

        self._thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="detector-thread"
        )
        self._thread.start()
        logger.info("Background detection thread started.")

    def submit_frame(self, frame: np.ndarray):
        """Submit a frame for detection — never blocks."""
        with self._frame_lock:
            self._pending_frame = frame.copy()

    def get_boxes(self) -> list:
        """Return latest cached boxes instantly — never blocks."""
        with self._boxes_lock:
            return list(self._boxes)

    def detect(self, frame: np.ndarray) -> list:
        """Legacy sync-style call — submits and returns cached boxes."""
        self.submit_frame(frame)
        return self.get_boxes()

    def stop(self):
        self._running = False

    def _detection_loop(self):
        """Runs forever in background thread."""
        while self._running:
            frame = None
            with self._frame_lock:
                if self._pending_frame is not None:
                    frame = self._pending_frame
                    self._pending_frame = None

            if frame is None:
                time.sleep(0.05)
                continue
            if self._model is None:
                time.sleep(0.1)
                continue

            try:
                input_frame = cv2.resize(frame, (320, 320))
                t0 = time.perf_counter()
                with self._model_lock:
                    results = self._model(
                        input_frame,
                        imgsz=320,
                        conf=Config.CONFIDENCE_THRESHOLD,
                        iou=Config.IOU_THRESHOLD,
                        max_det=Config.MAX_DETECTIONS,
                        verbose=False,
                        device=Config.DEVICE,
                    )
                ms = (time.perf_counter() - t0) * 1000
                boxes = self._parse_results(results, frame.shape)

                with self._boxes_lock:
                    self._boxes = boxes

                self.last_inference_ms = ms
                self.last_detection_count = len(boxes)
                self.last_labels = list({b["label"] for b in boxes})

            except Exception as e:
                logger.error("Detection error: %s", e)
                time.sleep(0.1)

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = Path(Config.MODEL_PATH)
            if model_path.exists():
                self._model = YOLO(str(model_path))
            else:
                logger.info("Downloading %s ...", Config.MODEL_NAME)
                self._model = YOLO(Config.MODEL_NAME)
            # Warm-up pass
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._model(dummy, imgsz=320, verbose=False, device=Config.DEVICE)
            logger.info("YOLOv8n loaded and warmed up.")
        except Exception as e:
            logger.error("Model load error: %s", e)

    def _parse_results(self, results, original_shape: tuple) -> list:
        h, w = original_shape[:2]
        sx, sy = w / 320, h / 320
        boxes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names.get(cls_id, str(cls_id))
                boxes.append({
                    "x1": int(xyxy[0] * sx), "y1": int(xyxy[1] * sy),
                    "x2": int(xyxy[2] * sx), "y2": int(xyxy[3] * sy),
                    "label": label, "confidence": conf,
                    "class_id": cls_id, "color": _class_color(cls_id),
                })
        return boxes
