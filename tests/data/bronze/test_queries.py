from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.queries import load_daily_price_state_by_ticker
from swingtrader.data.bronze.schema import metadata as bronze_metadata
from swingtrader.data.bronze.writer import upsert_daily_prices
from swingtrader.data.clients.yfinance import DAILY_PRICE_COLUMNS


def test_load_daily_price_state_by_ticker_returns_coverage_by_ticker() -> None:
    engine = _sqlite_engine()
    fetched_at = datetime(2026, 7, 4, 9, 30, tzinfo=UTC)
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="AAK.ST",
            close=278.2,
            fetched_at=fetched_at,
            request_id="test-request",
        ),
        engine=engine,
    )
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="BOL.ST",
            close=279.1,
            fetched_at=fetched_at,
            request_id="test-request",
        ),
        engine=engine,
    )

    states = load_daily_price_state_by_ticker(
        engine=engine,
        provider="yfinance",
        tickers=("AAK.ST", "BOL.ST", "MISSING.ST"),
    )

    assert set(states) == {"AAK.ST", "BOL.ST"}
    assert states["AAK.ST"].row_count == 1
    assert states["AAK.ST"].first_trading_date == date(2026, 6, 26)
    assert states["AAK.ST"].last_trading_date == date(2026, 6, 26)


def test_load_daily_price_state_by_ticker_returns_empty_for_no_tickers() -> None:
    engine = _sqlite_engine()

    states = load_daily_price_state_by_ticker(
        engine=engine,
        provider="yfinance",
        tickers=(),
    )

    assert states == {}


def _sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    bronze_metadata.create_all(engine)
    return engine


def _daily_prices(
    *,
    ticker: str,
    close: float,
    fetched_at: object,
    request_id: object,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "yfinance",
                "ticker": ticker,
                "trading_date": date(2026, 6, 26),
                "open": 275.4,
                "high": 279.0,
                "low": 274.8,
                "close": close,
                "adjusted_close": None,
                "volume": 124500,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "fetched_at": fetched_at,
                "request_id": request_id,
            }
        ],
        columns=DAILY_PRICE_COLUMNS,
    )
