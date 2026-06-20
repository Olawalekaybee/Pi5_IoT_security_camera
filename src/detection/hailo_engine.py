"""
Thin wrapper around HailoRT's VDevice API. Falls back to a CPU mock
backend automatically when no Hailo device is present, so the project
runs (with synthetic detections) on a laptop for demo/testing purposes.
"""

from __future__ import annotations
import logging
import time
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning(
        "hailo_platform not found — running in CPU MOCK mode. "
        "Install HailoRT and the hailo_platform Python package on the Pi "
        "to enable real NPU inference."
    )


_shared_vdevice = None


def _get_shared_vdevice():
    """
    A physical Hailo-8L exposes exactly one device. Both the detection
    model and the Re-ID model need to run on it, so we open the VDevice
    once and configure multiple network groups (one per .hef) on it,
    instead of each engine opening its own VDevice (which fails with
    HAILO_OUT_OF_PHYSICAL_DEVICES on the second call).
    """
    global _shared_vdevice
    if _shared_vdevice is None:
        _shared_vdevice = hpf.VDevice()
        logger.info("Opened shared Hailo VDevice")
    return _shared_vdevice


def release_shared_vdevice() -> None:
    """Call once at process shutdown, after all engines are closed."""
    global _shared_vdevice
    if _shared_vdevice is not None:
        try:
            _shared_vdevice.release()
        except Exception:
            pass
        _shared_vdevice = None


class HailoInferenceEngine:
    """
    Loads a compiled .hef model onto the Hailo-8L NPU and exposes a
    simple `infer(frame) -> raw_output` call. When Hailo hardware/SDK
    is unavailable, transparently swaps in a CPU mock that returns
    plausible synthetic detections so the rest of the pipeline (zones,
    Re-ID, alerting, dashboard) can be exercised end-to-end.
    """

    def __init__(self, hef_path: str, input_shape: Tuple[int, int] = (640, 640)):
        self.hef_path = hef_path
        self.input_shape = input_shape
        self.mock_mode = not HAILO_AVAILABLE
        self._device = None
        self._network_group = None
        self._infer_pipeline = None
        self._activation_context = None
        self._last_frame_shape = None

        if self.mock_mode:
            logger.info(f"[MOCK] HailoInferenceEngine ready (model: {hef_path})")
        else:
            self._load_hailo_model()

    def _load_hailo_model(self) -> None:
        """Load .hef onto the shared NPU device as its own network group."""
        try:
            self._device = _get_shared_vdevice()
            hef = hpf.HEF(self.hef_path)
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef, interface=hpf.HailoStreamInterface.PCIe
            )
            self._network_group = self._device.configure(hef, configure_params)[0]
            self._input_vstream_info = hef.get_input_vstream_infos()[0]
            self._output_vstream_info = hef.get_output_vstream_infos()[0]
            logger.info(f"Loaded HEF '{self.hef_path}' onto Hailo-8L NPU")
        except Exception as exc:
            logger.error(f"Failed to load Hailo model, falling back to mock: {exc}")
            self.mock_mode = True

    def infer(self, frame: np.ndarray) -> np.ndarray:
        """Run one inference pass. Returns raw model output tensor."""
        self._last_frame_shape = frame.shape[:2]  # (height, width) for coordinate scaling
        if self.mock_mode:
            return self._mock_infer(frame)
        return self._hailo_infer(frame)

    def _hailo_infer(self, frame: np.ndarray) -> np.ndarray:
        resized = self._preprocess(frame)

        if self._infer_pipeline is None:
            # The network group must be explicitly activated before any
            # vstream can be written to — configure() alone only defines
            # it. We keep the activation context alive for the engine's
            # lifetime alongside the cached InferVStreams pipeline.
            self._activation_context = self._network_group.activate()
            self._activation_context.__enter__()

            self._infer_pipeline = hpf.InferVStreams(
                self._network_group,
                hpf.InputVStreamParams.make(self._network_group),
                hpf.OutputVStreamParams.make(self._network_group),
            ).__enter__()

        input_data = {self._input_vstream_info.name: np.expand_dims(resized, axis=0)}
        results = self._infer_pipeline.infer(input_data)
        raw = results[self._output_vstream_info.name]

        # Detection models (e.g. YOLOv8) use on-chip NMS-by-class output;
        # embedding models (e.g. OSNet Re-ID) output a flat feature
        # vector. Branch on the vstream's declared format rather than
        # assuming every model on this engine is a detector.
        if self._is_nms_output():
            return self._parse_nms_output(raw)
        return np.asarray(raw).flatten()

    def _is_nms_output(self) -> bool:
        """True if this HEF's output vstream is a Hailo NMS-by-class op
        (detection models), False for plain feature-vector outputs
        (embedding/Re-ID models)."""
        info = self._output_vstream_info
        format_type = getattr(getattr(info, "format", None), "order", None)
        type_name = str(format_type) if format_type is not None else ""
        return "NMS" in type_name or "nms" in self.hef_path.lower() and "yolo" in self.hef_path.lower()

    def _parse_nms_output(self, raw) -> np.ndarray:
        """
        This HEF's output op is 'HAILO NMS BY CLASS': HailoRT returns a
        per-class list of arrays, each row [y1, x1, y2, x2, score] in
        normalized 0-1 coordinates. We flatten this into the pipeline's
        expected [x1, y1, x2, y2, conf, class_id] pixel-space format.
        """
        detections = []
        # raw is typically a list/array indexed by class id; batch dim
        # may already be squeezed depending on HailoRT version.
        per_class = raw[0] if len(raw) == 1 and hasattr(raw[0], "__len__") and not isinstance(raw[0], (int, float)) else raw

        # BUG FIX: normalized coords must be scaled by the ORIGINAL
        # frame's height/width, not the model's fixed square input size
        # (e.g. 640x640). Using input_shape here silently produced
        # out-of-bounds boxes whenever the source frame wasn't square
        # (a 640x480 camera frame got y-coordinates scaled as if the
        # frame were 640 tall, overflowing past the real 480px height).
        frame_h, frame_w = getattr(self, "_last_frame_shape", (self.input_shape[1], self.input_shape[0]))

        for class_id, class_dets in enumerate(per_class):
            if class_dets is None or len(class_dets) == 0:
                continue
            for det in class_dets:
                if len(det) < 5:
                    continue
                y1, x1, y2, x2, score = det[:5]
                detections.append((
                    x1 * frame_w, y1 * frame_h,
                    x2 * frame_w, y2 * frame_h,
                    score, class_id,
                ))

        if not detections:
            return np.empty((0, 6))
        return np.array(detections)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        This HEF expects UINT8 NHWC input (not normalized float) — Hailo
        bakes normalization into the compiled model itself. Sending
        float32 0-1 data here would silently produce garbage detections.
        """
        import cv2
        resized = cv2.resize(frame, self.input_shape)
        return resized.astype(np.uint8)

    def _mock_infer(self, frame: np.ndarray) -> np.ndarray:
        """
        Synthetic output: occasionally 'detects' a person-shaped box
        roughly where motion-like brightness variance is highest, so
        the demo behaves a bit like a real feed rather than pure noise.
        """
        time.sleep(0.01)  # simulate ~10ms inference latency
        h, w = frame.shape[:2]
        if np.random.random() < 0.3:
            cx, cy = np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4)
            bw, bh = w // 6, h // 3
            return np.array([[
                max(0, cx - bw // 2), max(0, cy - bh // 2),
                min(w, cx + bw // 2), min(h, cy + bh // 2),
                np.random.uniform(0.55, 0.95),  # confidence
                0,                                # class id: person
            ]])
        return np.empty((0, 6))

    def close(self) -> None:
        """
        Releases this engine's network group only. The shared VDevice
        itself is released once via release_shared_vdevice() at process
        shutdown — not here, since other engines may still be using it.
        """
        if self._infer_pipeline is not None:
            try:
                self._infer_pipeline.__exit__(None, None, None)
            except Exception:
                pass
        if self._activation_context is not None:
            try:
                self._activation_context.__exit__(None, None, None)
            except Exception:
                pass
        if self._network_group is not None:
            try:
                self._network_group.shutdown()
            except Exception:
                pass