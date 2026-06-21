"""
SQLite-backed event store. Uses WAL mode for safe concurrent
reads from the dashboard while the detection pipeline writes.
"""

from __future__ import annotations
import sqlite3
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvent:
    """A single detection event flowing through the pipeline."""
    timestamp: float = field(default_factory=time.time)
    zone: str = "unknown"
    bbox: tuple = (0, 0, 0, 0)
    detection_confidence: float = 0.0
    crop: Any = None                       # numpy array, not persisted
    person_id: Optional[str] = None
    reid_confidence: float = 0.0
    snapshot_path: Optional[str] = None
    alerted: bool = False
    liveness_static: Optional[bool] = None  # True if motion heuristic flagged a likely photo/screen spoof


class Database:
    """Thread-safe SQLite wrapper for event logging."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        zone TEXT NOT NULL,
        bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
        detection_confidence REAL,
        person_id TEXT,
        reid_confidence REAL,
        snapshot_path TEXT,
        alerted INTEGER DEFAULT 0,
        liveness_static INTEGER DEFAULT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone);
    CREATE INDEX IF NOT EXISTS idx_events_alerted ON events(alerted);

    CREATE TABLE IF NOT EXISTS zone_cooldowns (
        zone TEXT PRIMARY KEY,
        last_alert_ts REAL NOT NULL
    );
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        """Each thread gets its own connection (SQLite requirement)."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                self.db_path, check_same_thread=False, timeout=10
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL;")
            self._local.conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def init_tables(self) -> None:
        with self._lock:
            self.conn.executescript(self.SCHEMA)
            self.conn.commit()
            self._migrate_add_liveness_column()
        logger.info(f"Database initialized at {self.db_path} (WAL mode)")

    def _migrate_add_liveness_column(self) -> None:
        """
        CREATE TABLE IF NOT EXISTS won't add a new column to a database
        file that already exists from before this field was introduced —
        add it via ALTER TABLE if missing, so existing event databases
        (like one already running on a deployed Pi) upgrade cleanly
        instead of crashing on the new column reference.
        """
        existing_cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(events)")}
        if "liveness_static" not in existing_cols:
            self.conn.execute("ALTER TABLE events ADD COLUMN liveness_static INTEGER DEFAULT NULL")
            self.conn.commit()
            logger.info("Migrated events table: added liveness_static column")

    def insert_event(self, event: DetectionEvent) -> int:
        liveness_value = None if event.liveness_static is None else int(event.liveness_static)
        with self._lock:
            cur = self.conn.execute(
                """INSERT INTO events
                   (timestamp, zone, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    detection_confidence, person_id, reid_confidence,
                    snapshot_path, alerted, liveness_static)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.timestamp, event.zone,
                    *event.bbox,
                    event.detection_confidence,
                    event.person_id, event.reid_confidence,
                    event.snapshot_path, int(event.alerted),
                    liveness_value,
                ),
            )
            self.conn.commit()
            return cur.lastrowid

    def get_recent_events(self, limit: int = 50, zone: Optional[str] = None) -> List[dict]:
        query = "SELECT * FROM events"
        params: tuple = ()
        if zone:
            query += " WHERE zone = ?"
            params = (zone,)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params = params + (limit,)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict:
        """
        Three separate small queries, each able to use a covering index
        (idx_events_alerted, idx_events_timestamp), measured faster on a
        24k-row table than a single combined query — a combined query
        with CASE WHEN sums forces SQLite into a full table scan since
        it needs every row's values, while these three each resolve
        from an index alone without touching table data.
        """
        total = self.conn.execute("SELECT COUNT(*) AS c FROM events").fetchone()["c"]
        alerts = self.conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE alerted = 1"
        ).fetchone()["c"]
        last_24h = self.conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE timestamp > ?",
            (time.time() - 86400,),
        ).fetchone()["c"]
        return {"total_events": total, "total_alerts": alerts, "events_24h": last_24h}

    def check_cooldown(self, zone: str, cooldown_seconds: int) -> bool:
        """Returns True if this zone is allowed to alert again."""
        with self._lock:
            row = self.conn.execute(
                "SELECT last_alert_ts FROM zone_cooldowns WHERE zone = ?", (zone,)
            ).fetchone()
            now = time.time()
            if row and (now - row["last_alert_ts"]) < cooldown_seconds:
                return False
            self.conn.execute(
                "INSERT INTO zone_cooldowns (zone, last_alert_ts) VALUES (?, ?) "
                "ON CONFLICT(zone) DO UPDATE SET last_alert_ts = ?",
                (zone, now, now),
            )
            self.conn.commit()
            return True

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()