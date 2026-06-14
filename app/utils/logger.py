"""Centralised rotating file + console logger."""

import logging
import os
from logging.handlers import RotatingFileHandler
from app.config import Config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(Config.LOG_LEVEL)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    log_path = Config.LOG_FILE
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = RotatingFileHandler(log_path, maxBytes=5_242_880, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
