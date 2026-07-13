from sqlalchemy import create_engine, inspect

from swingtrader.data.bronze.schema import BRONZE_MARKET_DAILY_PRICES_TABLE
from swingtrader.data.db import initialize_database, resolve_database_engine


def test_initialize_database_creates_bronze_market_daily_prices_table() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    initialize_database(engine)

    assert BRONZE_MARKET_DAILY_PRICES_TABLE in inspect(engine).get_table_names()


def test_initialize_database_is_idempotent() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    initialize_database(engine)
    initialize_database(engine)

    assert inspect(engine).get_table_names().count(BRONZE_MARKET_DAILY_PRICES_TABLE) == 1


def test_resolve_database_engine_initializes_data_schema_by_default() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    resolved_engine = resolve_database_engine(engine=engine)

    assert resolved_engine is engine
    assert BRONZE_MARKET_DAILY_PRICES_TABLE in inspect(engine).get_table_names()


def test_resolve_database_engine_can_skip_schema_initialization() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    resolved_engine = resolve_database_engine(engine=engine, initialize=False)

    assert resolved_engine is engine
    assert BRONZE_MARKET_DAILY_PRICES_TABLE not in inspect(engine).get_table_names()


def test_resolve_database_engine_rejects_engine_and_database_url() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    try:
        resolve_database_engine(database_url="sqlite+pysqlite:///unused.sqlite", engine=engine)
    except ValueError as error:
        assert str(error) == "Pass either engine or database_url, not both."
    else:
        raise AssertionError("Expected ValueError for duplicate engine configuration")
