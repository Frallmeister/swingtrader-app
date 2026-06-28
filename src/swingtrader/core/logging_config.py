"""Logging configuration helpers for runnable project entrypoints."""

import logging
import sys
from typing import TextIO

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: TextIO | None = None,
    force: bool = False,
) -> None:
    """Configure application logging for jobs, scripts, and notebooks.

    Library modules should create loggers with ``logging.getLogger(__name__)`` and
    should not call this function at import time. Runnable entrypoints should call
    it once before invoking application code.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        root_logger.setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        stream=stream or sys.stderr,
        force=force,
    )
