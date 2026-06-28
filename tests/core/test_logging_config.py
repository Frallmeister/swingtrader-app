import io
import logging
from collections.abc import Iterator

import pytest

from swingtrader.core.logging_config import configure_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    yield

    root_logger.handlers[:] = original_handlers
    root_logger.setLevel(original_level)


def test_configure_logging_writes_expected_format() -> None:
    stream = io.StringIO()

    configure_logging(stream=stream, force=True)
    logging.getLogger("swingtrader.test").info("download complete")

    output = stream.getvalue()

    assert "INFO swingtrader.test download complete" in output


def test_configure_logging_accepts_string_level() -> None:
    configure_logging(level="DEBUG", stream=io.StringIO(), force=True)

    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_does_not_add_duplicate_handlers() -> None:
    configure_logging(stream=io.StringIO(), force=True)
    root_logger = logging.getLogger()
    handler_count = len(root_logger.handlers)

    configure_logging()

    assert len(root_logger.handlers) == handler_count
