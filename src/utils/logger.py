"""Logging configuration for the security monitor."""

import logging
import sys
from pathlib import Path


def setup_logger(level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure root logger with console + rotating file output."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            Path(log_dir) / "monitor.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        handlers.append(file_handler)
    except OSError:
        pass

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
