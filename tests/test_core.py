"""
Unit tests — runnable on any machine (no Hailo hardware needed).
Run: pytest tests/ -v
"""

import numpy as np
import pytest

from src.utils.event import DetectionEvent
from src.utils.zones import ZoneChecker
from src.reid.identifier import cosine_similarity, l2_normalize
from src.utils.database import Database


# ------------------------------------------------------------------
# DetectionEvent
# ------------------------------------------------------------------

class TestDetectionEvent:
    def test_repr(self):
        e = DetectionEvent(
            timestamp="2025-01-01T12:00:00",
            bbox=(10, 20, 100, 200),
            confidence=0.88,
            zone="zone_a",
        )
        assert "zone_a" in repr(e)
        assert "0.88" in repr(e)


# ------------------------------------------------------------------
# ZoneChecker
# ------------------------------------------------------------------

class TestZoneChecker:
    POLYGONS = {
        "left":  [[0, 0], [320, 0], [320, 480], [0, 480]],
        "right": [[320, 0], [640, 0], [640, 480], [320, 480]],
    }

    def setup_method(self):
        self.checker = ZoneChecker(self.POLYGONS)

    def test_point_in_left_zone(self):
        assert self.checker.get_zone((100, 100)) == "left"

    def test_point_in_right_zone(self):
        assert self.checker.get_zone((500, 240)) == "right"

    def test_point_outside_all_zones(self):
        # Above the defined area
        assert self.checker.get_zone((320, 700)) is None

    def test_empty_polygons(self):
        checker = ZoneChecker({})
        assert checker.get_zone((100, 100)) is None


# ------------------------------------------------------------------
# Re-ID math
# ------------------------------------------------------------------

class TestReidMath:
    def test_l2_normalize_unit_vector(self):
        v = np.array([3.0, 4.0])
        n = l2_normalize(v)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6

    def test_cosine_similarity_identical(self):
        v = l2_normalize(np.random.randn(512))
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        a = l2_normalize(np.array([1.0, 0.0]))
        b = l2_normalize(np.array([0.0, 1.0]))
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_cosine_similarity_opposite(self):
        v = l2_normalize(np.random.randn(512))
        assert cosine_similarity(v, -v) < -0.99


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------

class TestDatabase:
    def setup_method(self, tmp_path=None):
        self.db = Database(":memory:")
        self.db.init_tables()

    def _make_event(self, zone="zone_a", person_id=None):
        return DetectionEvent(
            timestamp="2025-01-01T12:00:00",
            bbox=(10, 20, 100, 200),
            confidence=0.85,
            zone=zone,
            person_id=person_id,
            reid_confidence=0.7,
        )

    def test_insert_and_retrieve(self):
        self.db.insert_event(self._make_event())
        events = self.db.get_recent_events(limit=10)
        assert len(events) == 1
        assert events[0]["zone"] == "zone_a"

    def test_stats_unknown_count(self):
        self.db.insert_event(self._make_event(person_id=None))
        self.db.insert_event(self._make_event(person_id="alice"))
        stats = self.db.get_stats()
        assert stats["total_events"] == 2
        assert stats["unknown_count"] == 1

    def test_filter_by_zone(self):
        self.db.insert_event(self._make_event(zone="zone_a"))
        self.db.insert_event(self._make_event(zone="zone_b"))
        events = self.db.get_events_by_zone("zone_a")
        assert all(e["zone"] == "zone_a" for e in events)
