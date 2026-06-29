"""Application configuration helpers."""

import os
from pathlib import Path

DATABASE_URL_ENV_VAR = "SWINGTRADER_DATABASE_URL"
DEFAULT_SQLITE_DATABASE_PATH = Path("data") / "swingtrader.sqlite"
DEFAULT_DATABASE_URL = f"sqlite+pysqlite:///{DEFAULT_SQLITE_DATABASE_PATH.as_posix()}"


def get_database_url(database_url: str | None = None) -> str:
    """Return the configured database URL, falling back to local SQLite."""
    if database_url:
        return database_url
    return os.environ.get(DATABASE_URL_ENV_VAR, DEFAULT_DATABASE_URL)
