"""
Configuration management — loads settings.yaml and provides typed access.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class ModelSettings:
    detection_hef: str = "models/yolov8n.hef"
    reid_hef: str = "models/resnet50_reid.hef"


@dataclass
class DatabaseSettings:
    path: str = "data/events.db"


@dataclass
class TelegramSettings:
    bot_token: str = ""
    chat_id: str = ""
    cooldown_seconds: int = 30    # Min seconds between alerts for same zone


@dataclass
class ReidSettings:
    similarity_threshold: float = 0.75   # Below this = unknown person
    alert_threshold: float = 0.60        # Extra-low confidence → alert anyway
    embedding_cache_size: int = 100


@dataclass
class ZoneSettings:
    restricted: List[str] = field(default_factory=lambda: ["zone_a", "zone_b"])
    polygons: dict = field(default_factory=dict)
    # Polygon format: {"zone_a": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ...}


@dataclass
class DashboardSettings:
    port: int = 5000
    host: str = "0.0.0.0"
    max_events_page: int = 50


@dataclass
class DetectionSettings:
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    input_width: int = 640
    input_height: int = 640
    target_fps: int = 30


@dataclass
class Settings:
    models: ModelSettings = field(default_factory=ModelSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    telegram: TelegramSettings = field(default_factory=TelegramSettings)
    reid: ReidSettings = field(default_factory=ReidSettings)
    zones: ZoneSettings = field(default_factory=ZoneSettings)
    dashboard: DashboardSettings = field(default_factory=DashboardSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    known_faces_dir: str = "data/known_faces"

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        yaml_path = Path(path)
        if not yaml_path.exists():
            import logging
            logging.getLogger(__name__).warning(
                f"Config file {path} not found — using defaults"
            )
            return cls()

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            models=ModelSettings(**data.get("models", {})),
            database=DatabaseSettings(**data.get("database", {})),
            telegram=TelegramSettings(**data.get("telegram", {})),
            reid=ReidSettings(**data.get("reid", {})),
            zones=ZoneSettings(**data.get("zones", {})),
            dashboard=DashboardSettings(**data.get("dashboard", {})),
            detection=DetectionSettings(**data.get("detection", {})),
            known_faces_dir=data.get("known_faces_dir", "data/known_faces"),
        )
