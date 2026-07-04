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

    Parameters
    ----------
    level
        Logging level for the root logger. Accepts the same integer or string values as
        ``logging.basicConfig``.
    stream
        Optional text stream for log output. When omitted, logs are written to ``sys.stderr``.
    force
        When ``True``, replace existing handlers using ``logging.basicConfig(force=True)``.
        When ``False`` and handlers already exist, only the root logger level is updated so
        repeated calls do not add duplicate handlers.
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
