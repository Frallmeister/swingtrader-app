from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert
from sqlalchemy.pool import StaticPool

from swingtrader.data.bronze.schema import bronze_market_daily_prices, metadata as bronze_metadata


@pytest.fixture
def seeded_engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    bronze_metadata.create_all(engine)
    rows = []
    first = date(2020, 1, 2)
    for day_number in range(8):
        trading_date = first + timedelta(days=day_number)
        if trading_date.weekday() >= 5:
            continue
        opening = 100.0 + day_number
        rows.append(
            {
                "provider": "yfinance",
                "ticker": "AAA.ST",
                "trading_date": trading_date,
                "open": opening,
                "high": opening + 2.0,
                "low": opening - 2.0,
                "close": opening + 1.0,
                "adjusted_close": opening + 1.0,
                "volume": 10_000,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "fetched_at": datetime.now(UTC),
                "request_id": "test",
            }
        )
    with engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), rows)
    return engine
