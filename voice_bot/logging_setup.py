"""Per-session logging.

Call :func:`setup_session_logging` once at startup. It creates a fresh, uniquely
named log file for the session under ``LOG_DIR`` and wires the ``voice_bot``
logger to write every step there (calls, transcripts, replies, tool calls,
timings, errors). A separate file is created for each run.

Modules log via ``logging.getLogger(__name__)`` (e.g. "voice_bot.pipeline"),
which propagates to the configured "voice_bot" logger.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from .config import settings


def setup_session_logging() -> Path:
    """Configure logging for this session and return the log file path."""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"session_{stamp}.log"

    logger = logging.getLogger("voice_bot")
    logger.setLevel(settings.log_level)
    logger.propagate = False
    # Reset handlers in case setup runs twice in one process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if settings.log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    logger.info("=== session start === log=%s", path)
    return path
