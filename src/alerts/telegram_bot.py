"""
Telegram alerting. Sends a message (with snapshot if available) when
an unrecognized person is detected in a restricted zone. Respects a
per-zone cooldown so a lingering intruder doesn't spam the chat.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class TelegramAlert:
    API_BASE = "https://api.telegram.org/bot{token}"

    def __init__(self, token: str, chat_id: str, enabled: bool = True,
                 cooldown_seconds: int = 30):
        self.token = token
        self.chat_id = chat_id
        self.cooldown_seconds = cooldown_seconds
        self.enabled = enabled and bool(token) and bool(chat_id) and REQUESTS_AVAILABLE
        self._last_alert_ts: dict[str, float] = {}

        if enabled and not self.enabled:
            logger.warning(
                "Telegram alerts requested but not fully configured "
                "(missing token/chat_id, or 'requests' not installed) — "
                "alerts will be logged only, not sent."
            )

    def _cooldown_ok(self, zone: str) -> bool:
        last = self._last_alert_ts.get(zone, 0)
        if time.time() - last < self.cooldown_seconds:
            return False
        self._last_alert_ts[zone] = time.time()
        return True

    def send(self, event) -> bool:
        if not self._cooldown_ok(event.zone):
            logger.debug(f"Skipping alert for zone '{event.zone}' — cooldown active")
            return False

        text = (
            f"🚨 Security alert\n"
            f"Zone: {event.zone}\n"
            f"Confidence: {event.detection_confidence:.0%}\n"
            f"Re-ID match: {'unknown person' if not event.person_id else event.person_id}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if not self.enabled:
            logger.info(f"[ALERT - not sent, disabled] {text}")
            event.alerted = False
            return False

        try:
            if event.snapshot_path and Path(event.snapshot_path).exists():
                self._send_photo(text, event.snapshot_path)
            else:
                self._send_text(text)
            event.alerted = True
            logger.info(f"Telegram alert sent for zone '{event.zone}'")
            return True
        except Exception as exc:
            logger.error(f"Failed to send Telegram alert: {exc}")
            event.alerted = False
            return False

    def _send_text(self, text: str) -> None:
        url = self.API_BASE.format(token=self.token) + "/sendMessage"
        requests.post(url, data={"chat_id": self.chat_id, "text": text}, timeout=10)

    def _send_photo(self, caption: str, photo_path: str) -> None:
        url = self.API_BASE.format(token=self.token) + "/sendPhoto"
        with open(photo_path, "rb") as f:
            requests.post(
                url,
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": f},
                timeout=15,
            )
