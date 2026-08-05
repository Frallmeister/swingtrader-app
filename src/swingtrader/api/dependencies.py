"""FastAPI dependencies for application services."""

from functools import lru_cache

from swingtrader.replay.service import ReplayService


@lru_cache(maxsize=1)
def get_replay_service() -> ReplayService:
    return ReplayService()
