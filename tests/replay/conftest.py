from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert
from sqlalchemy.pool import StaticPool

from swingtrader.data.bronze.schema import bronze_market_daily_prices, metadata as bronze_metadata
from swingtrader.replay.service import ReplayService


@pytest.fixture
def engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    bronze_metadata.create_all(engine)
    return engine


@pytest.fixture
def seeded_engine(engine):
    rows = []
    first = date(2020, 1, 2)
    for day_number in range(8):
        trading_date = first + timedelta(days=day_number)
        if trading_date.weekday() >= 5:
            continue
        for ticker, offset in (("AAA.ST", 0.0), ("BBB.ST", 20.0)):
            opening = 100.0 + offset + day_number
            rows.append(
                {
                    "provider": "yfinance",
                    "ticker": ticker,
                    "trading_date": trading_date,
                    "open": opening,
                    "high": opening + 2.0,
                    "low": opening - 2.0,
                    "close": opening + 1.0,
                    "adjusted_close": opening + 1.0,
                    "volume": 10_000 + day_number,
                    "dividends": 0.0,
                    "stock_splits": 0.0,
                    "fetched_at": datetime.now(UTC),
                    "request_id": "test",
                }
            )
    with engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), rows)
    return engine


@pytest.fixture
def service(seeded_engine):
    return ReplayService(engine=seeded_engine)
