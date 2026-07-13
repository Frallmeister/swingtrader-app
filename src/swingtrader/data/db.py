"""Database initialization helpers for data schemas.

Use this module when callers need a ready-to-use data database. Generic SQLAlchemy engine
creation remains in ``swingtrader.core.db``; this module composes that infrastructure with
the currently implemented data schemas.
"""

from sqlalchemy.engine import Engine

from swingtrader.core.db import resolve_database_engine as resolve_core_database_engine
from swingtrader.data.bronze.schema import metadata as bronze_metadata


def initialize_database(engine: Engine) -> None:
    """Create known data tables if they do not already exist.

    Parameters
    ----------
    engine
        SQLAlchemy engine for the target database.

    Notes
    -----
    This is the application/data schema initialization entry point. Add future data schema
    metadata here rather than importing domain tables from ``swingtrader.core``.
    """
    bronze_metadata.create_all(engine)


def resolve_database_engine(
    *,
    database_url: str | None = None,
    engine: Engine | None = None,
    initialize: bool = True,
) -> Engine:
    """Return a data database engine, optionally initializing known data tables.

    Parameters
    ----------
    database_url
        Optional SQLAlchemy database URL. Mutually exclusive with ``engine``.
    engine
        Optional already-created SQLAlchemy engine. Useful for tests or callers that manage
        engine lifecycle themselves. Mutually exclusive with ``database_url``.
    initialize
        Whether to create known data tables before returning the engine.

    Returns
    -------
    Engine
        Existing or newly created SQLAlchemy engine.
    """
    resolved_engine = resolve_core_database_engine(database_url=database_url, engine=engine)
    if initialize:
        initialize_database(resolved_engine)
    return resolved_engine
