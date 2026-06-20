"""
Detection pipeline: pulls frames from the camera, runs YOLOv8 on the
Hailo NPU (or CPU mock), applies NMS, maps detections to zones, and
emits DetectionEvent callbacks. Also supports a benchmark mode that
reports FPS/latency without needing alerting or the dashboard.
"""

from __future__ import annotations
import logging
import time
import threading
from typing import Callable, Optional, List
import numpy as np

from src.detection.hailo_engine import HailoInferenceEngine
from src.utils.database import DetectionEvent
from src.utils.zones import resolve_zone

logger = logging.getLogger(__name__)


class DetectionPipeline:
    def __init__(self, model_path: str, camera_index, zones, settings):
        self.engine = HailoInferenceEngine(
            model_path,
            input_shape=(settings.detection.input_width, settings.detection.input_height),
        )
        self.camera_index = camera_index
        self.zones = zones
        self.settings = settings
        self.on_detection: Optional[Callable[[DetectionEvent], None]] = None
        self._running = False
        self._cap = None
        self._camera_backend = None

    def _open_camera(self):
        """
        Tries picamera2 first (the correct API for CSI Pi Camera Modules
        on Bookworm/libcamera — cv2.VideoCapture can falsely report
        'opened' on these while never returning real frames). Falls
        back to OpenCV VideoCapture for USB webcams, and finally to
        synthetic frames if no camera is available at all.
        """
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            self._camera_backend = "picamera2"
            logger.info("Camera opened via picamera2 (CSI Pi Camera Module)")
            return picam2
        except Exception as exc:
            logger.info(f"picamera2 unavailable or no CSI camera ({exc}) — trying USB/OpenCV")

        import cv2
        cap = cv2.VideoCapture(self.camera_index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                self._camera_backend = "opencv"
                logger.info(f"Camera opened via OpenCV (index {self.camera_index})")
                return cap
            cap.release()

        logger.warning(
            f"Could not open any camera (CSI or index {self.camera_index}) — "
            "falling back to synthetic frames for demo purposes"
        )
        self._camera_backend = None
        return None

    def _read_frame(self) -> np.ndarray:
        if self._cap is not None:
            if self._camera_backend == "picamera2":
                # picamera2 returns RGB; the rest of the pipeline (and
                # OpenCV's own VideoCapture) assumes BGR, so convert to
                # keep color channels consistent across both backends.
                import cv2
                rgb_frame = self._cap.capture_array()
                return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            elif self._camera_backend == "opencv":
                ok, frame = self._cap.read()
                if ok:
                    return frame
        # Synthetic frame fallback (no camera attached)
        return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    def _nms(self, detections: np.ndarray) -> List[tuple]:
        """Filter detections by confidence threshold (NMS already applied
        on-chip by the compiled HEF in production; this is the CPU-side
        confidence gate for both real and mock paths)."""
        results = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if conf >= self.settings.detection.confidence_threshold:
                results.append((int(x1), int(y1), int(x2), int(y2), float(conf)))
        return results

    def run(self) -> None:
        self._running = True
        self._cap = self._open_camera()
        logger.info("Detection pipeline started")
        frame_interval = 1.0 / self.settings.detection.target_fps

        while self._running:
            t0 = time.time()
            frame = self._read_frame()
            raw_output = self.engine.infer(frame)
            detections = self._nms(raw_output)

            for (x1, y1, x2, y2, conf) in detections:
                zone = resolve_zone((x1, y1, x2, y2), self.zones.polygons)
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                event = DetectionEvent(
                    zone=zone,
                    bbox=(x1, y1, x2, y2),
                    detection_confidence=conf,
                    crop=crop,
                )
                if self.on_detection:
                    self.on_detection(event)

            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run_benchmark(self, duration_s: int = 30) -> None:
        """Standalone benchmark mode: prints FPS/latency without alerting/dashboard."""
        self._cap = self._open_camera()
        logger.info(f"Running benchmark for {duration_s}s...")
        latencies = []
        start = time.time()
        frame_count = 0

        while time.time() - start < duration_s:
            frame = self._read_frame()
            t0 = time.time()
            self.engine.infer(frame)
            latencies.append((time.time() - t0) * 1000)
            frame_count += 1

        total_time = time.time() - start
        avg_latency = sum(latencies) / len(latencies)
        fps = frame_count / total_time

        print("\n=== Benchmark results ===")
        print(f"Mode:            {'CPU MOCK' if self.engine.mock_mode else 'Hailo-8L NPU'}")
        print(f"Frames:          {frame_count}")
        print(f"Duration:        {total_time:.1f}s")
        print(f"Avg FPS:         {fps:.1f}")
        print(f"Avg latency:     {avg_latency:.2f} ms")
        print(f"Min/Max latency: {min(latencies):.2f} / {max(latencies):.2f} ms")

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            try:
                if self._camera_backend == "picamera2":
                    self._cap.stop()
                elif self._camera_backend == "opencv":
                    self._cap.release()
            except Exception:
                pass
        self.engine.close()
        logger.info("Detection pipeline stopped")