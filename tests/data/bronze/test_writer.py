from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from swingtrader.data.bronze import writer as bronze_writer
from swingtrader.data.bronze.schema import bronze_market_daily_prices
from swingtrader.data.bronze.schema import metadata as bronze_metadata
from swingtrader.data.bronze.writer import (
    BRONZE_MARKET_DAILY_PRICE_COLUMNS,
    upsert_daily_prices,
)


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    bronze_metadata.create_all(engine)
    return engine


def test_upsert_daily_prices_inserts_new_rows(sqlite_engine: Engine) -> None:
    prices = _daily_prices(close=278.2, request_id="first-request")

    written_rows = upsert_daily_prices(prices=prices, engine=sqlite_engine)

    with sqlite_engine.connect() as connection:
        row = connection.execute(select(bronze_market_daily_prices)).mappings().one()

    assert written_rows == 1
    assert row["provider"] == "yfinance"
    assert row["ticker"] == "AAK.ST"
    assert row["trading_date"] == date(2026, 6, 26)
    assert float(row["close"]) == 278.2
    assert row["request_id"] == "first-request"


def test_upsert_daily_prices_updates_existing_rows(sqlite_engine: Engine) -> None:
    first_fetched_at = datetime(2026, 6, 28, 10, 30, tzinfo=UTC)
    second_fetched_at = datetime(2026, 6, 28, 11, 0, tzinfo=UTC)

    upsert_daily_prices(
        prices=_daily_prices(close=278.2, fetched_at=first_fetched_at, request_id="first-request"),
        engine=sqlite_engine,
    )
    written_rows = upsert_daily_prices(
        prices=_daily_prices(close=279.1, fetched_at=second_fetched_at, request_id="rerun-request"),
        engine=sqlite_engine,
    )

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))
        row = connection.execute(select(bronze_market_daily_prices)).mappings().one()

    assert written_rows == 1
    assert row_count == 1
    assert float(row["close"]) == 279.1
    assert row["fetched_at"] == second_fetched_at.replace(tzinfo=None)
    assert row["request_id"] == "rerun-request"


def test_upsert_daily_prices_handles_empty_dataframe(sqlite_engine: Engine) -> None:
    prices = pd.DataFrame(columns=BRONZE_MARKET_DAILY_PRICE_COLUMNS)

    written_rows = upsert_daily_prices(prices=prices, engine=sqlite_engine)

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))

    assert written_rows == 0
    assert row_count == 0


def test_upsert_daily_prices_rejects_missing_columns(sqlite_engine: Engine) -> None:
    prices = pd.DataFrame([{"provider": "yfinance"}])

    with pytest.raises(ValueError, match="Missing bronze daily price columns"):
        upsert_daily_prices(prices=prices, engine=sqlite_engine)


def test_calculate_sqlite_chunk_size_uses_safety_factor() -> None:
    chunk_size = bronze_writer._calculate_sqlite_chunk_size(
        max_variables=32766,
        column_count=13,
    )

    assert chunk_size == 2268


def test_upsert_daily_prices_inserts_multiple_chunks(sqlite_engine: Engine, monkeypatch) -> None:
    monkeypatch.setattr(bronze_writer, "_upsert_chunk_size", _forced_chunk_size(2))
    prices = _daily_price_rows(row_count=5, close=278.2, request_id="chunked-request")

    written_rows = upsert_daily_prices(prices=prices, engine=sqlite_engine)

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))

    assert written_rows == 5
    assert row_count == 5


def test_upsert_daily_prices_updates_existing_rows_across_chunks(
    sqlite_engine: Engine,
    monkeypatch,
) -> None:
    monkeypatch.setattr(bronze_writer, "_upsert_chunk_size", _forced_chunk_size(2))
    first_fetched_at = datetime(2026, 6, 28, 10, 30, tzinfo=UTC)
    second_fetched_at = datetime(2026, 6, 28, 11, 0, tzinfo=UTC)

    upsert_daily_prices(
        prices=_daily_price_rows(
            row_count=5,
            close=278.2,
            fetched_at=first_fetched_at,
            request_id="first-request",
        ),
        engine=sqlite_engine,
    )
    written_rows = upsert_daily_prices(
        prices=_daily_price_rows(
            row_count=5,
            close=279.1,
            fetched_at=second_fetched_at,
            request_id="rerun-request",
        ),
        engine=sqlite_engine,
    )

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))
        rows = connection.execute(select(bronze_market_daily_prices)).mappings().all()

    assert written_rows == 5
    assert row_count == 5
    assert {float(row["close"]) for row in rows} == {279.1}
    assert {row["fetched_at"] for row in rows} == {second_fetched_at.replace(tzinfo=None)}
    assert {row["request_id"] for row in rows} == {"rerun-request"}


def test_upsert_daily_prices_rolls_back_all_chunks_on_failure(
    sqlite_engine: Engine,
    monkeypatch,
) -> None:
    monkeypatch.setattr(bronze_writer, "_upsert_chunk_size", _forced_chunk_size(1))
    prices = _daily_price_rows(row_count=3, close=278.2, request_id="rollback-request")
    prices.loc[1, "request_id"] = None

    with pytest.raises(IntegrityError):
        upsert_daily_prices(prices=prices, engine=sqlite_engine)

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))

    assert row_count == 0


def _daily_prices(
    *,
    close: float,
    fetched_at: datetime | None = None,
    request_id: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "yfinance",
                "ticker": "AAK.ST",
                "trading_date": date(2026, 6, 26),
                "open": 275.4,
                "high": 279.0,
                "low": 274.8,
                "close": close,
                "adjusted_close": None,
                "volume": 124500,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "fetched_at": fetched_at or datetime(2026, 6, 28, 10, 30, tzinfo=UTC),
                "request_id": request_id,
            }
        ]
    )


def _daily_price_rows(
    *,
    row_count: int,
    close: float,
    fetched_at: datetime | None = None,
    request_id: str,
) -> pd.DataFrame:
    base_row = _daily_prices(
        close=close,
        fetched_at=fetched_at,
        request_id=request_id,
    ).iloc[0]
    rows = []
    for day_offset in range(row_count):
        row = base_row.to_dict()
        row["trading_date"] = date(2026, 6, 26) + timedelta(days=day_offset)
        rows.append(row)
    return pd.DataFrame(rows)


def _forced_chunk_size(chunk_size: int):
    def _chunk_size(**_kwargs) -> int:
        return chunk_size

    return _chunk_size
