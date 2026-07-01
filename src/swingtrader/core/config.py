"""Configuration helpers for resolving application database settings."""

import os
from pathlib import Path

DATABASE_URL_ENV_VAR = "SWINGTRADER_DATABASE_URL"
DEFAULT_SQLITE_DATABASE_PATH = Path("data") / "swingtrader.sqlite"
DEFAULT_DATABASE_URL = f"sqlite+pysqlite:///{DEFAULT_SQLITE_DATABASE_PATH.as_posix()}"


def get_database_url(database_url: str | None = None) -> str:
    """Return the database URL used by application storage helpers.

    Parameters
    ----------
    database_url
        Explicit SQLAlchemy database URL supplied by the caller. When provided, this value
        takes precedence over environment configuration.

    Returns
    -------
    str
        The resolved SQLAlchemy database URL. Resolution order is explicit ``database_url``,
        ``SWINGTRADER_DATABASE_URL``, then the default local SQLite database.
    """
    if database_url:
        return database_url
    return os.environ.get(DATABASE_URL_ENV_VAR, DEFAULT_DATABASE_URL)
