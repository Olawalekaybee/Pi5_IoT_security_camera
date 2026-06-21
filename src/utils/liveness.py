"""
Lightweight, single-RGB-camera liveness heuristic.

HONEST SCOPE: this is NOT a production-grade anti-spoofing system. Real
liveness detection (Face ID-style) uses dedicated IR/depth/structured-light
hardware that this Pi + single camera doesn't have. What this module does
instead is a motion-based heuristic: it tracks how much a person's
bounding box (and optionally a downsampled crop region) changes across
consecutive frames. A held-up photo or phone screen is perfectly rigid
— it moves only as much as the holder's hand shakes, uniformly, as one
flat plane. A real person standing in frame has small, non-uniform
natural motion: postural sway, breathing, blinking, weight shifting.

This catches the common, low-effort spoofing case from the project's
own testing (holding up a phone photo) but will NOT reliably catch a
photo mounted on a steady tripod, a printed photo physically wiggled by
hand, or any genuinely sophisticated replay attack. It is a heuristic
signal, not a security guarantee — treat it as a "probably suspicious"
flag, not a hard pass/fail gate, especially at this early stage.
"""

from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
import numpy as np


@dataclass
class _Track:
    positions: Deque[Tuple[float, float]]  # box center history
    sizes: Deque[Tuple[float, float]]       # box width/height history
    last_seen: float


class LivenessTracker:
    """
    Tracks bounding-box motion per recognized identity across frames to
    flag suspiciously static (likely photo/screen) detections.

    Usage: call update(person_id, bbox) once per frame per detection
    that has a person_id (only known/enrolled people, since that's the
    case that actually matters — an unknown person already triggers an
    alert regardless of liveness).
    """

    def __init__(
        self,
        history_len: int = 15,
        min_samples: int = 8,
        motion_threshold_px: float = 1.2,
        track_timeout_s: float = 5.0,
    ):
        """
        history_len: how many recent frames to keep per identity.
        min_samples: minimum frames seen before making any liveness call
            (avoids false "suspicious" flags on a person who just walked in).
        motion_threshold_px: average frame-to-frame center movement, in
            pixels, below which a track is flagged as suspiciously static.
            Tuned conservatively low — real people rarely sit phone-still;
            adjust upward if you see false positives on calm/seated subjects.
        track_timeout_s: drop a track if not updated for this long, so
            stale identities don't linger forever in memory.
        """
        self.history_len = history_len
        self.min_samples = min_samples
        self.motion_threshold_px = motion_threshold_px
        self.track_timeout_s = track_timeout_s
        self._tracks: Dict[str, _Track] = {}

    def _bbox_center_and_size(self, bbox: Tuple[int, int, int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), (x2 - x1, y2 - y1)

    def update(self, person_id: str, bbox: Tuple[int, int, int, int]) -> None:
        """Record this frame's detection for the given identity."""
        center, size = self._bbox_center_and_size(bbox)
        now = time.time()

        track = self._tracks.get(person_id)
        if track is None:
            track = _Track(
                positions=deque(maxlen=self.history_len),
                sizes=deque(maxlen=self.history_len),
                last_seen=now,
            )
            self._tracks[person_id] = track

        track.positions.append(center)
        track.sizes.append(size)
        track.last_seen = now
        self._prune_stale(now)

    def _prune_stale(self, now: float) -> None:
        stale = [pid for pid, t in self._tracks.items() if now - t.last_seen > self.track_timeout_s]
        for pid in stale:
            del self._tracks[pid]

    def get_liveness_score(self, person_id: str) -> Optional[float]:
        """
        Returns average frame-to-frame center displacement in pixels, or
        None if not enough history yet to make a call. Higher = more
        natural motion observed; values near zero suggest a static
        photo/screen rather than a live person.
        """
        track = self._tracks.get(person_id)
        if track is None or len(track.positions) < self.min_samples:
            return None

        positions = list(track.positions)
        displacements = [
            float(np.hypot(positions[i][0] - positions[i - 1][0],
                            positions[i][1] - positions[i - 1][1]))
            for i in range(1, len(positions))
        ]
        return float(np.mean(displacements))

    def is_suspiciously_static(self, person_id: str) -> Optional[bool]:
        """
        Returns True if this identity's recent motion is below the
        static-photo heuristic threshold, False if motion looks natural,
        or None if there isn't enough history yet to judge. None should
        be treated as "no opinion yet" — don't flag or clear on it.
        """
        score = self.get_liveness_score(person_id)
        if score is None:
            return None
        return score < self.motion_threshold_px

    def reset(self, person_id: Optional[str] = None) -> None:
        """Clear tracking history for one identity, or all if None."""
        if person_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(person_id, None)