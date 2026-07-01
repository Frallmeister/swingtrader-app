"""Database engine and schema helpers."""

from pathlib import Path

from sqlalchemy import create_engine, make_url
from sqlalchemy.engine import Engine

from swingtrader.core.config import get_database_url
from swingtrader.data.bronze.schema import metadata


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the application database.

    Parameters
    ----------
    database_url
        Optional SQLAlchemy database URL. When omitted, the URL is resolved from
        application configuration by ``get_database_url``.

    Returns
    -------
    Engine
        SQLAlchemy engine connected to the resolved database URL.

    Notes
    -----
    For file-backed SQLite URLs, the parent directory is created before the engine is
    constructed. In-memory SQLite and non-SQLite URLs are left untouched.
    """
    resolved_database_url = get_database_url(database_url)
    _ensure_sqlite_parent_directory(resolved_database_url)
    return create_engine(resolved_database_url)


def initialize_database(engine: Engine) -> None:
    """Create known application tables if they do not already exist.

    Parameters
    ----------
    engine
        SQLAlchemy engine for the target database.

    Notes
    -----
    This uses the current application metadata directly. It is intended for local and early
    application setup; a future migration layer can replace it when schema management grows.
    """
    metadata.create_all(engine)


def _ensure_sqlite_parent_directory(database_url: str) -> None:
    url = make_url(database_url)
    database_path = url.database
    if not url.drivername.startswith("sqlite") or database_path is None:
        return
    if database_path in ("", ":memory:"):
        return

    database_parent = Path(database_path).parent
    if database_parent != Path(""):
        database_parent.mkdir(parents=True, exist_ok=True)
