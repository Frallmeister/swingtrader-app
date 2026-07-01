"""Writers for bronze market data tables."""

from collections.abc import Iterable

import pandas as pd
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.schema import bronze_market_daily_prices

BRONZE_MARKET_DAILY_PRICE_COLUMNS = tuple(
    column.name for column in bronze_market_daily_prices.columns
)
BRONZE_MARKET_DAILY_PRICE_PRIMARY_KEY = ("provider", "ticker", "trading_date")
BRONZE_MARKET_DAILY_PRICE_UPDATE_COLUMNS = tuple(
    column
    for column in BRONZE_MARKET_DAILY_PRICE_COLUMNS
    if column not in BRONZE_MARKET_DAILY_PRICE_PRIMARY_KEY
)


def upsert_daily_prices(prices: pd.DataFrame, engine: Engine) -> int:
    """Insert or update bronze daily price rows by provider, ticker, and date.

    Parameters
    ----------
    prices
        DataFrame containing the bronze daily price columns defined by
        ``bronze_market_daily_prices``. Extra columns are ignored; missing required columns
        raise ``ValueError``.
    engine
        SQLAlchemy engine for the destination database. SQLite and PostgreSQL are supported.

    Returns
    -------
    int
        Number of input rows submitted to the upsert statement. Empty DataFrames are treated
        as a no-op and return ``0``.

    Raises
    ------
    ValueError
        Raised when required bronze columns are missing or when the engine dialect is not
        supported for idempotent upserts.

    Notes
    -----
    Rows are matched on ``provider``, ``ticker``, and ``trading_date``. On conflict, all
    market value columns plus ``fetched_at`` and ``request_id`` are updated from the incoming
    row.
    """
    _validate_daily_price_columns(prices.columns)
    if prices.empty:
        return 0

    rows = _to_database_rows(prices.loc[:, BRONZE_MARKET_DAILY_PRICE_COLUMNS])
    statement = _build_upsert_statement(engine=engine, rows=rows)
    with engine.begin() as connection:
        connection.execute(statement)
    return len(rows)


def _validate_daily_price_columns(columns: Iterable[str]) -> None:
    missing_columns = sorted(set(BRONZE_MARKET_DAILY_PRICE_COLUMNS) - set(columns))
    if missing_columns:
        msg = f"Missing bronze daily price columns: {', '.join(missing_columns)}"
        raise ValueError(msg)


def _to_database_rows(prices: pd.DataFrame) -> list[dict[str, object]]:
    # pandas represent missing values as e.g. NaN and NaT. Databases expect None with SQLAlchemy.
    database_prices = prices.astype(object).where(pd.notna(prices), None)
    return list(database_prices.to_dict(orient="records"))


def _build_upsert_statement(engine: Engine, rows: list[dict[str, object]]):
    if engine.dialect.name == "sqlite":
        return _build_sqlite_upsert_statement(rows)
    if engine.dialect.name == "postgresql":
        return _build_postgresql_upsert_statement(rows)

    msg = f"Unsupported database dialect for bronze upsert: {engine.dialect.name}"
    raise ValueError(msg)


def _build_sqlite_upsert_statement(rows: list[dict[str, object]]):
    from sqlalchemy.dialects.sqlite import insert

    statement = insert(bronze_market_daily_prices).values(rows)
    update_values = _build_update_values(statement)
    return statement.on_conflict_do_update(
        index_elements=BRONZE_MARKET_DAILY_PRICE_PRIMARY_KEY,
        set_=update_values,
    )


def _build_postgresql_upsert_statement(rows: list[dict[str, object]]):
    from sqlalchemy.dialects.postgresql import insert

    statement = insert(bronze_market_daily_prices).values(rows)
    update_values = _build_update_values(statement)
    return statement.on_conflict_do_update(
        index_elements=BRONZE_MARKET_DAILY_PRICE_PRIMARY_KEY,
        set_=update_values,
    )


def _build_update_values(statement):
    update_values = {
        column: getattr(statement.excluded, column)
        for column in BRONZE_MARKET_DAILY_PRICE_UPDATE_COLUMNS
    }
    return update_values
