"""Unit tests for the SQLite event database — no hardware required."""

import pytest
import tempfile
import os
from src.utils.database import Database, DetectionEvent


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    database.init_tables()
    yield database
    database.close()
    os.unlink(path)


class TestDatabase:
    def test_insert_and_retrieve_event(self, db):
        event = DetectionEvent(zone="zone_a", bbox=(1, 2, 3, 4), detection_confidence=0.9)
        row_id = db.insert_event(event)
        assert row_id > 0

        events = db.get_recent_events(limit=10)
        assert len(events) == 1
        assert events[0]["zone"] == "zone_a"

    def test_filter_by_zone(self, db):
        db.insert_event(DetectionEvent(zone="zone_a"))
        db.insert_event(DetectionEvent(zone="zone_b"))

        zone_a_events = db.get_recent_events(zone="zone_a")
        assert len(zone_a_events) == 1
        assert zone_a_events[0]["zone"] == "zone_a"

    def test_stats_counts(self, db):
        db.insert_event(DetectionEvent(zone="zone_a", alerted=True))
        db.insert_event(DetectionEvent(zone="zone_a", alerted=False))

        stats = db.get_stats()
        assert stats["total_events"] == 2
        assert stats["total_alerts"] == 1

    def test_cooldown_blocks_repeated_alerts(self, db):
        assert db.check_cooldown("zone_a", cooldown_seconds=60) is True
        assert db.check_cooldown("zone_a", cooldown_seconds=60) is False

    def test_cooldown_independent_per_zone(self, db):
        assert db.check_cooldown("zone_a", cooldown_seconds=60) is True
        assert db.check_cooldown("zone_b", cooldown_seconds=60) is True
