from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import create_engine, insert

from swingtrader.data.bronze.queries import (
    load_daily_price_quality_state_by_ticker,
    load_daily_price_state_by_ticker,
)
from swingtrader.data.bronze.schema import bronze_market_daily_prices, metadata


def test_bronze_state_and_quality_queries_apply_inclusive_end_date() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    rows = [
        _row(date(2026, 1, 1), close=10, volume=100),
        _row(date(2026, 1, 2), close=20, volume=100),
        _row(date(2026, 1, 3), close=1000, volume=100),
    ]
    with engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), rows)

    coverage = load_daily_price_state_by_ticker(
        engine=engine,
        provider="test",
        tickers=("AAA",),
        end_date=date(2026, 1, 2),
    )["AAA"]
    quality = load_daily_price_quality_state_by_ticker(
        engine=engine,
        provider="test",
        tickers=("AAA",),
        turnover_lookback_rows=2,
        end_date=date(2026, 1, 2),
    )["AAA"]

    assert coverage.row_count == 2
    assert coverage.last_trading_date == date(2026, 1, 2)
    assert quality.row_count == 2
    assert quality.latest_median_turnover == Decimal("1500")


def _row(trading_date: date, *, close: int, volume: int) -> dict[str, object]:
    return {
        "provider": "test",
        "ticker": "AAA",
        "trading_date": trading_date,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adjusted_close": close,
        "volume": volume,
        "dividends": 0,
        "stock_splits": 0,
        "fetched_at": datetime(2026, 1, 4),
        "request_id": "test-request",
    }
