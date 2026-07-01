from pathlib import Path

from sqlalchemy import inspect

from swingtrader.core.config import DATABASE_URL_ENV_VAR, get_database_url
from swingtrader.core.db import create_database_engine, initialize_database
from swingtrader.data.bronze.schema import BRONZE_MARKET_DAILY_PRICES_TABLE


def test_get_database_url_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV_VAR, "sqlite+pysqlite:///from-env.sqlite")

    database_url = get_database_url("sqlite+pysqlite:///explicit.sqlite")

    assert database_url == "sqlite+pysqlite:///explicit.sqlite"


def test_get_database_url_reads_environment_override(monkeypatch) -> None:
    monkeypatch.setenv(DATABASE_URL_ENV_VAR, "sqlite+pysqlite:///from-env.sqlite")

    database_url = get_database_url()

    assert database_url == "sqlite+pysqlite:///from-env.sqlite"


def test_create_database_engine_creates_sqlite_parent_and_initializes_schema(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "nested" / "swingtrader.sqlite"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"

    engine = create_database_engine(database_url)
    initialize_database(engine)

    assert database_path.parent.is_dir()
    assert BRONZE_MARKET_DAILY_PRICES_TABLE in inspect(engine).get_table_names()
