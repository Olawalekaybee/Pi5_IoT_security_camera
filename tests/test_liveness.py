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
        """A perfectly unmoving bbox (like a held-up photo) should score zero speed."""
        tracker = LivenessTracker(min_samples=5, motion_threshold_px_per_sec=3.0)
        for _ in range(10):
            tracker.update("alice", (100, 100, 200, 300))  # identical every sample

        score = tracker.get_liveness_score("alice")
        assert score == 0.0
        assert tracker.is_suspiciously_static("alice") is True

    def test_naturally_moving_bbox_not_flagged(self):
        """Small realistic jitter, sampled at a typical per-frame rate, should
        NOT be flagged static."""
        tracker = LivenessTracker(min_samples=5, motion_threshold_px_per_sec=3.0)
        positions = [
            (100, 100, 200, 300), (103, 101, 203, 301), (98, 99, 198, 299),
            (105, 104, 205, 304), (101, 98, 201, 298), (99, 102, 199, 302),
            (104, 100, 204, 300), (97, 97, 197, 297),
        ]
        for bbox in positions:
            tracker.update("alice", bbox)
            time.sleep(0.01)  # simulate ~100fps-ish per-sample spacing

        score = tracker.get_liveness_score("alice")
        assert score is not None and score > 3.0
        assert tracker.is_suspiciously_static("alice") is False

    def test_throttled_sampling_rate_does_not_falsely_flag(self):
        """
        Regression test for a real integration bug: when update() is fed by
        an upstream throttle that only samples a given person every ~0.4s
        (instead of every camera frame), the same per-sample pixel jitter
        used to look "slower" under a fixed-threshold-per-sample scheme even
        though the person moved the same amount — because more real time
        passed between samples. Scoring in px/sec (this test) should NOT
        be sensitive to the sampling interval the way a raw per-sample
        distance metric would be.
        """
        tracker = LivenessTracker(min_samples=5, motion_threshold_px_per_sec=3.0)
        positions = [
            (100, 100, 200, 300), (108, 103, 208, 303), (97, 96, 197, 296),
            (110, 107, 210, 307), (101, 95, 201, 295), (96, 104, 196, 304),
        ]
        for bbox in positions:
            tracker.update("alice", bbox)
            time.sleep(0.4)  # simulate throttled ~2.5 samples/sec spacing

        score = tracker.get_liveness_score("alice")
        # Real, natural-looking movement should still register as live
        # motion even though samples arrived far apart in time.
        assert score is not None and score > 3.0
        assert tracker.is_suspiciously_static("alice") is False

    def test_independent_tracks_per_identity(self):
        tracker = LivenessTracker(min_samples=3, motion_threshold_px_per_sec=3.0)
        for _ in range(5):
            tracker.update("alice", (100, 100, 200, 300))  # static
            time.sleep(0.01)
        for i in range(5):
            tracker.update("bob", (100 + i * 10, 100, 200 + i * 10, 300))  # moving
            time.sleep(0.01)

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
        assert len(track.timestamps) == 5

    def test_zero_or_negative_dt_samples_are_skipped(self):
        """If two updates land on the exact same timestamp (e.g. clock
        resolution or test environment quirks), that pair shouldn't produce
        a divide-by-zero or an artificially infinite speed."""
        tracker = LivenessTracker(min_samples=2, motion_threshold_px_per_sec=3.0)
        tracker.update("alice", (100, 100, 200, 300))
        tracker.update("alice", (150, 150, 250, 350))  # near-instant second call
        # Should not raise, and should return a finite (possibly None) score
        score = tracker.get_liveness_score("alice")
        assert score is None or (score >= 0 and score < float("inf"))