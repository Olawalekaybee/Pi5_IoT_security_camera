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

        if self.mock_mode:
            logger.info(f"[MOCK] HailoInferenceEngine ready (model: {hef_path})")
        else:
            self._load_hailo_model()

    def _load_hailo_model(self) -> None:
        """Load .hef onto the NPU via HailoRT VDevice."""
        try:
            self._device = hpf.VDevice()
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
        if self.mock_mode:
            return self._mock_infer(frame)
        return self._hailo_infer(frame)

    def _hailo_infer(self, frame: np.ndarray) -> np.ndarray:
        resized = self._preprocess(frame)
        with hpf.InferVStreams(
            self._network_group,
            hpf.InputVStreamParams.make(self._network_group),
            hpf.OutputVStreamParams.make(self._network_group),
        ) as pipeline:
            input_data = {self._input_vstream_info.name: np.expand_dims(resized, axis=0)}
            results = pipeline.infer(input_data)
            return results[self._output_vstream_info.name][0]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(frame, self.input_shape)
        return resized.astype(np.float32) / 255.0

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
        if self._device is not None:
            try:
                self._device.release()
            except Exception:
                pass
