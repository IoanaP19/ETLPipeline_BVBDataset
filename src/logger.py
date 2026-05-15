"""
src/logger.py
-------------
Central logging configuration for the BVB ETL pipeline.

Creates a logger that writes to TWO destinations simultaneously:
  1. Console  (INFO level  — you see progress while the pipeline runs)
  2. Log file (DEBUG level — full detail saved to outputs/pipeline.log)

Usage in any module:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")
    log.warning("something suspicious")
    log.error("something failed")
"""

import logging
import os
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger attached to the shared pipeline handlers.
    Call once per module: log = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything at the logger level

    # ── Handler 1: Console ────────────────────────────────────────────────────
    # Shows INFO and above — clean, readable output while the pipeline runs.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    ))

    # ── Handler 2: File ───────────────────────────────────────────────────────
    # Shows DEBUG and above — full trace saved for review after the run.
    log_path = Path(__file__).resolve().parent.parent / "outputs" / "pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
