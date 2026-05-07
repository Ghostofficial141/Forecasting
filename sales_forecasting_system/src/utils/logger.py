"""
logger.py
=========
Production-grade logging configuration for the Sales Forecasting System.
Implements structured logging with both file and console handlers,
rotating file handler to prevent disk bloat, and consistent formatting.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
BACKUP_COUNT = 5                # keep 5 rotated files


def get_logger(
    name: str,
    level: int = logging.DEBUG,
    log_to_file: bool = True,
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Factory function that returns a fully configured logger.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : int
        Logging level (default: DEBUG).
    log_to_file : bool
        Whether to add a rotating file handler in addition to console.
    log_filename : str, optional
        Custom log filename. Defaults to '<name>_<date>.log'.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ---- Console Handler ----
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- File Handler ----
    if log_to_file:
        os.makedirs(LOG_DIR, exist_ok=True)

        if log_filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            log_filename = f"{name.replace('.', '_')}_{date_str}.log"

        log_path = os.path.join(LOG_DIR, log_filename)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


# Module-level default logger
logger = get_logger("forecasting_system")
