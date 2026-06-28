from datetime import UTC, date, datetime

import pytest
from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Numeric,
    String,
    create_engine,
    func,
    insert,
    inspect,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from swingtrader.data.bronze.schema import (
    BRONZE_MARKET_DAILY_PRICES_TABLE,
    bronze_market_daily_prices,
    metadata,
)


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    return engine


def test_bronze_market_daily_prices_schema_can_be_created_in_sqlite(
    sqlite_engine: Engine,
) -> None:
    inspector = inspect(sqlite_engine)

    columns = inspector.get_columns(BRONZE_MARKET_DAILY_PRICES_TABLE)
    column_names = [column["name"] for column in columns]

    assert column_names == [
        "provider",
        "ticker",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
        "fetched_at",
        "request_id",
    ]


def test_bronze_market_daily_prices_schema_defines_column_contract() -> None:
    columns = bronze_market_daily_prices.c

    assert [column.name for column in bronze_market_daily_prices.primary_key] == [
        "provider",
        "ticker",
        "trading_date",
    ]
    assert isinstance(columns.provider.type, String)
    assert isinstance(columns.ticker.type, String)
    assert isinstance(columns.trading_date.type, Date)
    assert isinstance(columns.volume.type, BigInteger)
    assert isinstance(columns.fetched_at.type, DateTime)
    assert isinstance(columns.request_id.type, String)

    for column_name in [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "dividends",
        "stock_splits",
    ]:
        column_type = columns[column_name].type
        assert isinstance(column_type, Numeric)
        assert column_type.precision == 18
        assert column_type.scale == 6


def test_bronze_market_daily_prices_rejects_duplicate_provider_ticker_date(
    sqlite_engine: Engine,
) -> None:
    row = {
        "provider": "yfinance",
        "ticker": "AAK.ST",
        "trading_date": date(2026, 6, 26),
        "open": 275.4,
        "high": 279.0,
        "low": 274.8,
        "close": 278.2,
        "adjusted_close": None,
        "volume": 124500,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "fetched_at": datetime(2026, 6, 28, 10, 30, tzinfo=UTC),
        "request_id": "first-request",
    }

    with sqlite_engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), row)

    duplicate_row = row | {
        "close": 279.1,
        "fetched_at": datetime(2026, 6, 28, 11, 0, tzinfo=UTC),
        "request_id": "rerun-request",
    }

    with pytest.raises(IntegrityError), sqlite_engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), duplicate_row)


def test_bronze_market_daily_prices_allows_same_ticker_date_from_different_providers(
    sqlite_engine: Engine,
) -> None:
    base_row = {
        "ticker": "AAK.ST",
        "trading_date": date(2026, 6, 26),
        "open": 275.4,
        "high": 279.0,
        "low": 274.8,
        "close": 278.2,
        "adjusted_close": None,
        "volume": 124500,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "fetched_at": datetime(2026, 6, 28, 10, 30, tzinfo=UTC),
    }
    rows = [
        base_row | {"provider": "yfinance", "request_id": "yfinance-request"},
        base_row | {"provider": "other_provider", "request_id": "other-request"},
    ]

    with sqlite_engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), rows)
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))

    assert row_count == 2
