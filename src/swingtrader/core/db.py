"""Database engine helpers.

These helpers centralize SQLAlchemy engine creation and generic engine resolution.
Domain-specific schema initialization lives outside ``core``.
"""

from pathlib import Path

from sqlalchemy import create_engine, make_url
from sqlalchemy.engine import Engine

from swingtrader.core.config import get_database_url


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


def resolve_database_engine(
    *,
    database_url: str | None = None,
    engine: Engine | None = None,
) -> Engine:
    """Return a SQLAlchemy engine from either an existing engine or database URL.

    Parameters
    ----------
    database_url
        Optional SQLAlchemy database URL. Mutually exclusive with ``engine``.
    engine
        Optional already-created SQLAlchemy engine. Useful for tests or callers that manage
        engine lifecycle themselves. Mutually exclusive with ``database_url``.

    Returns
    -------
    Engine
        Existing or newly created SQLAlchemy engine.
    """
    if engine is not None and database_url is not None:
        raise ValueError("Pass either engine or database_url, not both.")

    return engine or create_database_engine(database_url)


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
