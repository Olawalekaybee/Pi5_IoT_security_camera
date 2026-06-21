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

# BGR colors (OpenCV convention)
COLOR_KNOWN = (80, 200, 120)      # green-ish — recognized person
COLOR_UNKNOWN = (60, 60, 230)     # red-ish — unrecognized / alert
COLOR_ZONE_LINE = (200, 160, 60)  # amber — zone boundary overlay


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

        # Latest annotated (boxes + labels drawn) JPEG-encoded frame,
        # consumed by the dashboard's MJPEG stream. A lock protects it
        # since the Flask request thread and the pipeline thread both
        # touch it concurrently.
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()

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
            # create_preview_configuration is tuned for responsiveness/
            # quality, not throughput — it was measured adding ~65ms of
            # capture latency per frame on the IMX477 (vs ~10ms here).
            # create_video_configuration + an explicit FrameDurationLimits
            # forces the sensor into a fast, fixed-rate readout mode.
            config = picam2.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameDurationLimits": (8333, 8333)},  # ~120fps cap
            )
            picam2.configure(config)
            picam2.start()
            self._camera_backend = "picamera2"
            logger.info("Camera opened via picamera2 (CSI Pi Camera Module, video mode)")
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

    def _draw_zone_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw configured zone polygons as semi-transparent outlines,
        so the live feed shows restricted-area boundaries like real
        CCTV/VMS software."""
        import cv2
        for zone_name, polygon in self.zones.polygons.items():
            if len(polygon) < 3:
                continue
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            is_restricted = zone_name in self.zones.restricted
            color = COLOR_UNKNOWN if is_restricted else COLOR_ZONE_LINE
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
            label_pos = (int(polygon[0][0]) + 6, int(polygon[0][1]) + 18)
            cv2.putText(frame, zone_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color, 1, cv2.LINE_AA)
        return frame

    def _annotate_and_publish(self, frame: np.ndarray, annotated_dets: List[dict]) -> None:
        """
        Draws zone boundaries, bounding boxes, and person labels onto a
        copy of the frame, JPEG-encodes it, and stores it for the
        dashboard's MJPEG stream to pick up. annotated_dets entries:
        {bbox, detection_confidence, person_id, reid_confidence, zone, alerted}
        """
        import cv2
        canvas = frame.copy()
        canvas = self._draw_zone_overlays(canvas)

        for det in annotated_dets:
            x1, y1, x2, y2 = det["bbox"]
            known = det.get("person_id") is not None
            color = COLOR_KNOWN if known else COLOR_UNKNOWN

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            name = det.get("person_id") or "unknown"
            reid_conf = det.get("reid_confidence", 0.0)
            label = f"{name} ({reid_conf:.0%})" if known else f"{name} ({det['detection_confidence']:.0%})"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(0, y1 - 6)
            cv2.rectangle(canvas, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2), color, -1)
            cv2.putText(canvas, label, (x1 + 3, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._frame_lock:
                self._latest_jpeg = jpeg.tobytes()

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Thread-safe read of the most recent annotated frame, JPEG-encoded."""
        with self._frame_lock:
            return self._latest_jpeg

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

            frame_events = []
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
                    # on_detection (main.py's handle_detection) runs Re-ID
                    # and mutates event.person_id / event.reid_confidence
                    # in place before returning — we read those back below
                    # to annotate the video feed with the same identity
                    # the dashboard table and alerts are using.
                    self.on_detection(event)
                frame_events.append(event)

            self._annotate_and_publish(frame, [
                {
                    "bbox": ev.bbox,
                    "detection_confidence": ev.detection_confidence,
                    "person_id": ev.person_id,
                    "reid_confidence": ev.reid_confidence,
                }
                for ev in frame_events
            ])

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