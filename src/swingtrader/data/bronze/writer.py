"""Persist normalized market data into bronze tables.

Writer functions accept bronze-shaped DataFrames and use database upserts so reruns update
existing source rows instead of creating duplicates.
"""

from collections.abc import Iterable, Iterator
from math import floor

import pandas as pd
from sqlalchemy.engine import Connection, Engine

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
SQLITE_DEFAULT_MAX_VARIABLES = 999
SQLITE_VARIABLE_SAFETY_FACTOR = 0.9


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
    with engine.begin() as connection:
        chunk_size = _upsert_chunk_size(
            connection=connection,
            row_count=len(rows),
            column_count=len(BRONZE_MARKET_DAILY_PRICE_COLUMNS),
        )
        for chunk in _chunk_rows(rows=rows, chunk_size=chunk_size):
            statement = _build_upsert_statement(
                dialect_name=connection.dialect.name,
                rows=chunk,
            )
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


def _upsert_chunk_size(*, connection: Connection, row_count: int, column_count: int) -> int:
    if connection.dialect.name == "sqlite":
        return _calculate_sqlite_chunk_size(
            max_variables=_sqlite_max_variable_number(connection),
            column_count=column_count,
        )
    if connection.dialect.name == "postgresql":
        return row_count

    msg = f"Unsupported database dialect for bronze upsert: {connection.dialect.name}"
    raise ValueError(msg)


def _calculate_sqlite_chunk_size(*, max_variables: int, column_count: int) -> int:
    return max(1, floor(SQLITE_VARIABLE_SAFETY_FACTOR * max_variables / column_count))


def _sqlite_max_variable_number(connection: Connection) -> int:
    rows = connection.exec_driver_sql("PRAGMA compile_options").all()
    for row in rows:
        option = str(row[0])
        if option.startswith("MAX_VARIABLE_NUMBER="):
            try:
                return int(option.removeprefix("MAX_VARIABLE_NUMBER="))
            except ValueError:
                return SQLITE_DEFAULT_MAX_VARIABLES
    return SQLITE_DEFAULT_MAX_VARIABLES


def _chunk_rows(
    *,
    rows: list[dict[str, object]],
    chunk_size: int,
) -> Iterator[list[dict[str, object]]]:
    for start in range(0, len(rows), chunk_size):
        yield rows[start : start + chunk_size]


def _build_upsert_statement(*, dialect_name: str, rows: list[dict[str, object]]):
    if dialect_name == "sqlite":
        return _build_sqlite_upsert_statement(rows)
    if dialect_name == "postgresql":
        return _build_postgresql_upsert_statement(rows)

    msg = f"Unsupported database dialect for bronze upsert: {dialect_name}"
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
