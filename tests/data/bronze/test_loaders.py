from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.loaders import load_bronze_daily_prices
from swingtrader.data.bronze.schema import metadata as bronze_metadata
from swingtrader.data.bronze.writer import BRONZE_MARKET_DAILY_PRICE_COLUMNS, upsert_daily_prices
from swingtrader.data.market_frame import validate_market_price_index


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    bronze_metadata.create_all(engine)
    return engine


def test_load_bronze_daily_prices_loads_all_rows_for_provider(sqlite_engine: Engine) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(engine=sqlite_engine, provider="yfinance")

    assert list(prices.columns) == list(BRONZE_MARKET_DAILY_PRICE_COLUMNS)
    assert list(zip(prices["ticker"], prices["trading_date"], strict=True)) == [
        ("AAK.ST", pd.Timestamp("2026-06-24")),
        ("AAK.ST", pd.Timestamp("2026-06-25")),
        ("AAK.ST", pd.Timestamp("2026-06-26")),
        ("BOL.ST", pd.Timestamp("2026-06-25")),
        ("BOL.ST", pd.Timestamp("2026-06-26")),
    ]
    assert set(prices["provider"]) == {"yfinance"}


def test_load_bronze_daily_prices_supports_canonical_feature_index(sqlite_engine: Engine) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(engine=sqlite_engine, provider="yfinance")
    indexed = prices.set_index(["provider", "ticker", "trading_date"]).sort_index()

    validate_market_price_index(indexed)
    # The loader already returns rows in canonical order, so indexing without an
    # explicit sort produces the same canonical representation.
    pd.testing.assert_frame_equal(
        indexed,
        prices.set_index(["provider", "ticker", "trading_date"]),
    )


def test_load_bronze_daily_prices_filters_by_ticker(sqlite_engine: Engine) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        tickers=["BOL.ST"],
    )

    assert list(prices["ticker"]) == ["BOL.ST", "BOL.ST"]
    assert list(prices["trading_date"]) == [
        pd.Timestamp("2026-06-25"),
        pd.Timestamp("2026-06-26"),
    ]


def test_load_bronze_daily_prices_accepts_single_ticker_string(
    sqlite_engine: Engine,
) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        tickers="BOL.ST",
    )

    assert list(prices["ticker"]) == ["BOL.ST", "BOL.ST"]
    assert list(prices["trading_date"]) == [
        pd.Timestamp("2026-06-25"),
        pd.Timestamp("2026-06-26"),
    ]


def test_load_bronze_daily_prices_filters_by_start_date(sqlite_engine: Engine) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        start_date="2026-06-25",
    )

    assert list(prices["trading_date"]) == [
        pd.Timestamp("2026-06-25"),
        pd.Timestamp("2026-06-26"),
        pd.Timestamp("2026-06-25"),
        pd.Timestamp("2026-06-26"),
    ]


def test_load_bronze_daily_prices_filters_by_end_date(sqlite_engine: Engine) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        end_date=date(2026, 6, 25),
    )

    assert list(zip(prices["ticker"], prices["trading_date"], strict=True)) == [
        ("AAK.ST", pd.Timestamp("2026-06-24")),
        ("AAK.ST", pd.Timestamp("2026-06-25")),
        ("BOL.ST", pd.Timestamp("2026-06-25")),
    ]


def test_load_bronze_daily_prices_filters_by_ticker_and_date_range(
    sqlite_engine: Engine,
) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        tickers=["AAK.ST"],
        start_date="2026-06-25",
        end_date="2026-06-26",
    )

    assert list(prices["ticker"]) == ["AAK.ST", "AAK.ST"]
    assert list(prices["trading_date"]) == [
        pd.Timestamp("2026-06-25"),
        pd.Timestamp("2026-06-26"),
    ]


def test_load_bronze_daily_prices_selects_columns_with_keys(sqlite_engine: Engine) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        tickers=["AAK.ST"],
        columns=["close", "volume", "ticker", "close"],
    )

    assert list(prices.columns) == ["provider", "ticker", "trading_date", "close", "volume"]
    assert list(prices["ticker"]) == ["AAK.ST", "AAK.ST", "AAK.ST"]


def test_load_bronze_daily_prices_accepts_single_column_string(
    sqlite_engine: Engine,
) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        tickers="AAK.ST",
        columns="close",
    )

    assert list(prices.columns) == ["provider", "ticker", "trading_date", "close"]
    assert list(prices["ticker"]) == ["AAK.ST", "AAK.ST", "AAK.ST"]
    assert pd.api.types.is_float_dtype(prices["close"])


def test_load_bronze_daily_prices_returns_notebook_friendly_dtypes(
    sqlite_engine: Engine,
) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(engine=sqlite_engine, provider="yfinance")

    assert pd.api.types.is_datetime64_any_dtype(prices["trading_date"])
    assert str(prices["volume"].dtype) == "Int64"
    for column in [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "dividends",
        "stock_splits",
    ]:
        assert pd.api.types.is_float_dtype(prices[column])


def test_load_bronze_daily_prices_rejects_unknown_columns(sqlite_engine: Engine) -> None:
    with pytest.raises(ValueError, match="Unknown bronze daily price columns: missing_column"):
        load_bronze_daily_prices(
            engine=sqlite_engine,
            columns=["ticker", "missing_column"],
        )


def test_load_bronze_daily_prices_rejects_missing_table() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")

    with pytest.raises(ValueError, match="Missing bronze table: bronze_market_daily_prices"):
        load_bronze_daily_prices(engine=engine)


def test_load_bronze_daily_prices_returns_empty_dataframe_with_stable_columns(
    sqlite_engine: Engine,
) -> None:
    _seed_daily_prices(sqlite_engine)

    prices = load_bronze_daily_prices(
        engine=sqlite_engine,
        provider="yfinance",
        tickers=["MISSING.ST"],
        columns=["close", "volume"],
    )

    assert prices.empty
    assert list(prices.columns) == ["provider", "ticker", "trading_date", "close", "volume"]
    assert pd.api.types.is_datetime64_any_dtype(prices["trading_date"])
    assert pd.api.types.is_float_dtype(prices["close"])
    assert str(prices["volume"].dtype) == "Int64"


def _seed_daily_prices(engine: Engine) -> None:
    upsert_daily_prices(
        prices=pd.DataFrame(_daily_price_rows(), columns=BRONZE_MARKET_DAILY_PRICE_COLUMNS),
        engine=engine,
    )


def _daily_price_rows() -> list[dict[str, object]]:
    fetched_at = datetime(2026, 6, 28, 10, 30, tzinfo=UTC)
    return [
        _daily_price_row(
            provider="yfinance",
            ticker="BOL.ST",
            trading_date=date(2026, 6, 26),
            close=279.1,
            volume=225000,
            fetched_at=fetched_at,
        ),
        _daily_price_row(
            provider="yfinance",
            ticker="AAK.ST",
            trading_date=date(2026, 6, 26),
            close=278.2,
            volume=124500,
            fetched_at=fetched_at,
        ),
        _daily_price_row(
            provider="yfinance",
            ticker="AAK.ST",
            trading_date=date(2026, 6, 24),
            close=276.4,
            volume=103000,
            fetched_at=fetched_at,
        ),
        _daily_price_row(
            provider="other-provider",
            ticker="AAK.ST",
            trading_date=date(2026, 6, 26),
            close=281.7,
            volume=200000,
            fetched_at=fetched_at,
        ),
        _daily_price_row(
            provider="yfinance",
            ticker="BOL.ST",
            trading_date=date(2026, 6, 25),
            close=277.9,
            volume=180000,
            fetched_at=fetched_at,
        ),
        _daily_price_row(
            provider="yfinance",
            ticker="AAK.ST",
            trading_date=date(2026, 6, 25),
            close=277.0,
            volume=110000,
            fetched_at=fetched_at,
        ),
    ]


def _daily_price_row(
    *,
    provider: str,
    ticker: str,
    trading_date: date,
    close: float,
    volume: int,
    fetched_at: datetime,
) -> dict[str, object]:
    return {
        "provider": provider,
        "ticker": ticker,
        "trading_date": trading_date,
        "open": close - 1,
        "high": close + 1,
        "low": close - 2,
        "close": close,
        "adjusted_close": close,
        "volume": volume,
        "dividends": 0.0,
        "stock_splits": 0.0,
        "fetched_at": fetched_at,
        "request_id": "loader-test-request",
    }
