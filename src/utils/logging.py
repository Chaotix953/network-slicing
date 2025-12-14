# -*- coding: utf-8 -*-
"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional file path for logging
        format_string: Custom format string

    Returns:
        Root logger
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )

    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)
