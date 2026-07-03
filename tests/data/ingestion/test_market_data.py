from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.schema import bronze_market_daily_prices
from swingtrader.data.clients.yfinance import DAILY_PRICE_COLUMNS
from swingtrader.data.ingestion import market_data, universe_selection


@pytest.fixture
def sqlite_engine() -> Engine:
    return create_engine("sqlite+pysqlite:///:memory:")


def test_ingest_historical_daily_prices_writes_explicit_tickers(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_if_called() -> list[str]:
        raise AssertionError("active tickers should not be resolved for explicit ticker input")

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        ticker = _single_ticker(kwargs)
        calls.append(ticker)
        return _daily_prices(
            ticker=ticker,
            close=278.2,
            fetched_at=kwargs["fetched_at"],
            request_id=kwargs["request_id"],
        )

    monkeypatch.setattr(universe_selection, "resolve_active_tickers", fail_if_called)
    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = market_data.ingest_historical_daily_prices(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 29),
        tickers=[" AAK.ST ", "AAK.ST", "VOLV-B.ST"],
        engine=sqlite_engine,
        fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
        request_id="test-request",
    )

    with sqlite_engine.connect() as connection:
        stored_tickers = connection.scalars(
            select(bronze_market_daily_prices.c.ticker).order_by(
                bronze_market_daily_prices.c.ticker
            )
        ).all()

    assert calls == ["AAK.ST", "VOLV-B.ST"]
    assert result.tickers == ("AAK.ST", "VOLV-B.ST")
    assert result.downloaded_rows == 2
    assert result.upserted_rows == 2
    assert result.failures == ()
    assert stored_tickers == ["AAK.ST", "VOLV-B.ST"]


def test_ingest_historical_daily_prices_limits_active_tickers(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        ticker = _single_ticker(kwargs)
        calls.append(ticker)
        return pd.DataFrame(columns=DAILY_PRICE_COLUMNS)

    monkeypatch.setattr(
        universe_selection, "resolve_active_tickers", lambda config_dir=None: ["AAK.ST", "BOL.ST"]
    )
    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = market_data.ingest_historical_daily_prices(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 29),
        limit=1,
        engine=sqlite_engine,
    )

    assert calls == ["AAK.ST"]
    assert result.tickers == ("AAK.ST",)
    assert result.downloaded_rows == 0
    assert result.upserted_rows == 0


def test_ingest_historical_daily_prices_records_ticker_failures(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        ticker = _single_ticker(kwargs)
        if ticker == "FAIL.ST":
            raise RuntimeError("download failed")
        return _daily_prices(
            ticker=ticker,
            close=278.2,
            fetched_at=kwargs["fetched_at"],
            request_id=kwargs["request_id"],
        )

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = market_data.ingest_historical_daily_prices(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 29),
        tickers=["AAK.ST", "FAIL.ST"],
        engine=sqlite_engine,
        fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
        request_id="test-request",
    )

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))

    assert result.downloaded_rows == 1
    assert result.upserted_rows == 1
    assert len(result.failures) == 1
    assert result.failures[0].ticker == "FAIL.ST"
    assert result.failures[0].error_type == "RuntimeError"
    assert row_count == 1


def test_ingest_historical_daily_prices_can_raise_after_failures(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        raise RuntimeError(f"download failed for {_single_ticker(kwargs)}")

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    with pytest.raises(market_data.MarketDataIngestionError) as exc_info:
        market_data.ingest_historical_daily_prices(
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 29),
            tickers=["FAIL.ST"],
            engine=sqlite_engine,
            raise_on_failure=True,
        )

    assert exc_info.value.result.failures[0].ticker == "FAIL.ST"


def test_ingest_historical_daily_prices_is_idempotent(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closes = {"first-request": 278.2, "rerun-request": 279.1}

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        request_id = str(kwargs["request_id"])
        return _daily_prices(
            ticker=_single_ticker(kwargs),
            close=closes[request_id],
            fetched_at=kwargs["fetched_at"],
            request_id=request_id,
        )

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    for request_id in ["first-request", "rerun-request"]:
        market_data.ingest_historical_daily_prices(
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 29),
            tickers=["AAK.ST"],
            engine=sqlite_engine,
            fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
            request_id=request_id,
        )

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))
        row = connection.execute(select(bronze_market_daily_prices)).mappings().one()

    assert row_count == 1
    assert float(row["close"]) == 279.1
    assert row["request_id"] == "rerun-request"


def _single_ticker(kwargs: dict[str, object]) -> str:
    tickers = kwargs["tickers"]
    assert isinstance(tickers, list)
    assert len(tickers) == 1
    return str(tickers[0])


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
