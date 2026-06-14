"""
Basic smoke tests — run with: pytest tests/
"""

import pytest
import numpy as np


def test_detector_imports():
    """Detector module should import without ultralytics installed in test env."""
    try:
        from app.detection.detector import _class_color
        assert isinstance(_class_color(0), tuple)
        assert len(_class_color(0)) == 3
    except ImportError as e:
        pytest.skip(f"Optional import missing: {e}")


def test_class_color_wraps():
    from app.detection.detector import _class_color, _PALETTE
    for i in range(len(_PALETTE) + 5):
        color = _class_color(i)
        assert len(color) == 3


def test_config_defaults():
    from app.config import Config
    assert Config.CAMERA_WIDTH > 0
    assert Config.CAMERA_HEIGHT > 0
    assert 0 < Config.CONFIDENCE_THRESHOLD < 1
    assert Config.DETECTION_SKIP_FRAMES >= 1


def test_metrics_collector():
    from app.utils.metrics import MetricsCollector
    m = MetricsCollector(window=5)
    m.record_fps(25.0)
    m.record_fps(30.0)
    assert m.avg_fps == pytest.approx(27.5)

    m.record_inference(50.0)
    assert m.avg_inference_ms == pytest.approx(50.0)
