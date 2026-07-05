"""Pandas loaders for source-oriented bronze market data tables."""

from collections.abc import Sequence
from datetime import date

import pandas as pd
from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.schema import (
    BRONZE_MARKET_DAILY_PRICES_TABLE,
    bronze_market_daily_prices,
)

BRONZE_DAILY_PRICE_COLUMNS = tuple(column.name for column in bronze_market_daily_prices.columns)
BRONZE_DAILY_PRICE_KEY_COLUMNS = ("provider", "ticker", "trading_date")
BRONZE_DAILY_PRICE_FLOAT_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "dividends",
    "stock_splits",
)


def load_bronze_daily_prices(
    *,
    engine: Engine,
    provider: str = "yfinance",
    tickers: str | Sequence[str] | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    columns: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load bronze daily market prices into a pandas DataFrame.

    The loader is intended for notebook EDA and source data inspection. It applies provider,
    ticker, and date filters in SQL before rows are loaded into pandas. Date filters are
    inclusive: ``start_date`` means ``trading_date >= start_date`` and ``end_date`` means
    ``trading_date <= end_date``.

    When ``columns`` is omitted, all bronze daily price columns are returned in table order.
    When ``columns`` is provided, ``provider``, ``ticker``, and ``trading_date`` are always
    included first, followed by the requested non-key columns in caller order.

    ``trading_date`` is returned as pandas datetime, ``volume`` as nullable integer, and
    daily price/dividend/split columns as float for notebook-friendly analysis.

    Parameters
    ----------
    engine
        SQLAlchemy engine for the source database.
    provider
        Market data provider to filter by, such as ``"yfinance"``.
    tickers
        Optional provider ticker symbol or symbols to include. A single ticker may be passed
        as a string. An empty sequence returns an empty DataFrame with the expected columns.
    start_date
        Optional inclusive lower bound for ``trading_date``. Strings must use ISO date format.
    end_date
        Optional inclusive upper bound for ``trading_date``. Strings must use ISO date format.
    columns
        Optional bronze daily price column or columns to return. A single column may be
        passed as a string. Unknown columns raise ``ValueError``.

    Returns
    -------
    pandas.DataFrame
        Matching bronze daily price rows ordered by ``ticker`` and ``trading_date`` with
        notebook-friendly dtypes applied where relevant columns are present.

    Raises
    ------
    ValueError
        Raised for unknown column names, invalid date strings, or a missing bronze table.
    """
    selected_columns = _resolve_column_names(columns)
    selected_tickers = _normalize_string_sequence(tickers)
    if selected_tickers is not None and not selected_tickers:
        return pd.DataFrame(columns=selected_columns)

    _validate_table_exists(engine)

    table_columns = [bronze_market_daily_prices.c[column] for column in selected_columns]
    statement = select(*table_columns).where(bronze_market_daily_prices.c.provider == provider)

    if selected_tickers is not None:
        statement = statement.where(bronze_market_daily_prices.c.ticker.in_(selected_tickers))
    if start_date is not None:
        statement = statement.where(
            bronze_market_daily_prices.c.trading_date >= _parse_filter_date(start_date)
        )
    if end_date is not None:
        statement = statement.where(
            bronze_market_daily_prices.c.trading_date <= _parse_filter_date(end_date)
        )

    statement = statement.order_by(
        bronze_market_daily_prices.c.ticker,
        bronze_market_daily_prices.c.trading_date,
    )

    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()

    prices = pd.DataFrame.from_records(
        [dict(row) for row in rows],
        columns=selected_columns,
    )
    return _coerce_daily_price_dtypes(prices)


def _coerce_daily_price_dtypes(prices: pd.DataFrame) -> pd.DataFrame:
    coerced_prices = prices.copy()
    if "trading_date" in coerced_prices.columns:
        coerced_prices["trading_date"] = pd.to_datetime(coerced_prices["trading_date"])
    if "volume" in coerced_prices.columns:
        coerced_prices["volume"] = pd.to_numeric(coerced_prices["volume"]).astype("Int64")

    for column in BRONZE_DAILY_PRICE_FLOAT_COLUMNS:
        if column in coerced_prices.columns:
            coerced_prices[column] = pd.to_numeric(coerced_prices[column]).astype("float64")
    return coerced_prices


def _validate_table_exists(engine: Engine) -> None:
    if inspect(engine).has_table(BRONZE_MARKET_DAILY_PRICES_TABLE):
        return

    msg = (
        f"Missing bronze table: {BRONZE_MARKET_DAILY_PRICES_TABLE}. "
        "Initialize the application database with resolve_database_engine() or "
        "initialize_database(engine), or run a market data onboarding/update job first."
    )
    raise ValueError(msg)


def _resolve_column_names(columns: str | Sequence[str] | None) -> tuple[str, ...]:
    selected_columns = _normalize_string_sequence(columns)
    if selected_columns is None:
        return BRONZE_DAILY_PRICE_COLUMNS

    unknown_columns = sorted(set(selected_columns) - set(BRONZE_DAILY_PRICE_COLUMNS))
    if unknown_columns:
        msg = f"Unknown bronze daily price columns: {', '.join(unknown_columns)}"
        raise ValueError(msg)

    resolved_columns = [*BRONZE_DAILY_PRICE_KEY_COLUMNS]
    for column in selected_columns:
        if column not in BRONZE_DAILY_PRICE_KEY_COLUMNS and column not in resolved_columns:
            resolved_columns.append(column)
    return tuple(resolved_columns)


def _normalize_string_sequence(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _parse_filter_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        msg = f"Invalid date string: {value!r}. Expected ISO format YYYY-MM-DD."
        raise ValueError(msg) from error
