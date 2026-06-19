"""
DetectionEvent — the canonical data object passed between pipeline components.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import numpy as np


@dataclass
class DetectionEvent:
    timestamp: str
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2
    confidence: float
    zone: Optional[str] = None
    person_id: Optional[str] = None
    reid_confidence: float = 0.0
    crop: Optional[np.ndarray] = None    # BGR image crop
    frame: Optional[np.ndarray] = None  # Full frame (not stored to DB)
    crop_path: Optional[str] = None     # Path if crop saved to disk

    @classmethod
    def from_detection(cls, det, frame: np.ndarray) -> "DetectionEvent":
        """Build an event from a Detection + full frame."""
        return cls(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            bbox=det.bbox,
            confidence=det.confidence,
            zone=det.zone,
            crop=det.crop,
            frame=frame,
        )

    def __repr__(self):
        return (
            f"DetectionEvent(zone={self.zone!r}, conf={self.confidence:.2f}, "
            f"person_id={self.person_id!r}, reid={self.reid_confidence:.2f}, "
            f"ts={self.timestamp})"
        )
