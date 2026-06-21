"""Unit tests for the motion-based liveness heuristic — no hardware required."""

import pytest
import time
from src.utils.liveness import LivenessTracker


class TestLivenessTracker:
    def test_not_enough_samples_returns_none(self):
        tracker = LivenessTracker(min_samples=8)
        tracker.update("alice", (100, 100, 200, 300))
        assert tracker.get_liveness_score("alice") is None
        assert tracker.is_suspiciously_static("alice") is None

    def test_unknown_identity_returns_none(self):
        tracker = LivenessTracker()
        assert tracker.get_liveness_score("nobody") is None
        assert tracker.is_suspiciously_static("nobody") is None

    def test_static_bbox_flagged_as_suspicious(self):
        """A perfectly unmoving bbox (like a held-up photo) should score near zero."""
        tracker = LivenessTracker(min_samples=5, motion_threshold_px=1.2)
        for _ in range(10):
            tracker.update("alice", (100, 100, 200, 300))  # identical every frame

        score = tracker.get_liveness_score("alice")
        assert score == 0.0
        assert tracker.is_suspiciously_static("alice") is True

    def test_naturally_moving_bbox_not_flagged(self):
        """Small realistic jitter (a few px/frame) should NOT be flagged static."""
        tracker = LivenessTracker(min_samples=5, motion_threshold_px=1.2)
        positions = [
            (100, 100, 200, 300), (103, 101, 203, 301), (98, 99, 198, 299),
            (105, 104, 205, 304), (101, 98, 201, 298), (99, 102, 199, 302),
            (104, 100, 204, 300), (97, 97, 197, 297),
        ]
        for bbox in positions:
            tracker.update("alice", bbox)

        score = tracker.get_liveness_score("alice")
        assert score is not None and score > 1.2
        assert tracker.is_suspiciously_static("alice") is False

    def test_independent_tracks_per_identity(self):
        tracker = LivenessTracker(min_samples=3)
        for _ in range(5):
            tracker.update("alice", (100, 100, 200, 300))  # static
        for i in range(5):
            tracker.update("bob", (100 + i * 10, 100, 200 + i * 10, 300))  # moving

        assert tracker.is_suspiciously_static("alice") is True
        assert tracker.is_suspiciously_static("bob") is False

    def test_reset_single_identity(self):
        tracker = LivenessTracker(min_samples=3)
        for _ in range(5):
            tracker.update("alice", (100, 100, 200, 300))
        tracker.reset("alice")
        assert tracker.get_liveness_score("alice") is None

    def test_reset_all(self):
        tracker = LivenessTracker(min_samples=3)
        for _ in range(5):
            tracker.update("alice", (100, 100, 200, 300))
            tracker.update("bob", (50, 50, 150, 250))
        tracker.reset()
        assert tracker.get_liveness_score("alice") is None
        assert tracker.get_liveness_score("bob") is None

    def test_stale_track_pruned_after_timeout(self):
        tracker = LivenessTracker(min_samples=3, track_timeout_s=0.05)
        for _ in range(5):
            tracker.update("alice", (100, 100, 200, 300))
        assert tracker.get_liveness_score("alice") is not None

        time.sleep(0.1)
        tracker.update("bob", (0, 0, 10, 10))  # triggers pruning check
        assert "alice" not in tracker._tracks

    def test_history_respects_max_length(self):
        tracker = LivenessTracker(history_len=5, min_samples=3)
        for i in range(20):
            tracker.update("alice", (i, i, i + 100, i + 100))
        track = tracker._tracks["alice"]
        assert len(track.positions) == 5