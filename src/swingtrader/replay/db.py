"""Replay database initialization helpers."""

from sqlalchemy.engine import Engine

from swingtrader.core.db import resolve_database_engine
from swingtrader.replay.schema import metadata


def initialize_replay_database(engine: Engine) -> Engine:
    """Create replay tables that do not already exist and return the engine."""
    metadata.create_all(engine)
    return engine


def resolve_replay_database_engine(
    *, database_url: str | None = None, engine: Engine | None = None
) -> Engine:
    resolved = resolve_database_engine(database_url=database_url, engine=engine)
    return initialize_replay_database(resolved)
