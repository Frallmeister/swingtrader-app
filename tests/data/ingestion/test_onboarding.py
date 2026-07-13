from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine

import swingtrader.data.ingestion.onboarding as onboarding
from swingtrader.data.bronze.schema import bronze_market_daily_prices
from swingtrader.data.bronze.writer import upsert_daily_prices
from swingtrader.data.clients.yfinance import DAILY_PRICE_COLUMNS
from swingtrader.data.db import initialize_database
from swingtrader.data.ingestion.market_data import IngestionResult
from swingtrader.data.ingestion.onboarding import (
    BronzeOnboardingStatus,
    check_active_ticker_bronze_onboarding,
    sync_active_ticker_bronze_onboarding,
)


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    initialize_database(engine)
    return engine


@pytest.fixture
def universe_config_dir(tmp_path: Path) -> Path:
    _write_config(
        tmp_path / "se_large_cap.yml",
        """
kind: ticker_universe
list_name: se_large_cap
description: Test Swedish Large Cap universe
as_of_date: 2026-06-23
source: test
symbols:
  - ticker: VOLV-B.ST
    asset_type: EQUITY
  - ticker: AAK.ST
    asset_type: EQUITY
  - ticker: BOL.ST
    asset_type: EQUITY
""".lstrip(),
    )
    _write_config(
        tmp_path / "active_tickers.yml",
        """
kind: active_tickers
description: Test active ticker configuration
universes:
  - list_name: se_large_cap
    include: all
""".lstrip(),
    )
    return tmp_path


def test_check_active_ticker_bronze_onboarding_classifies_active_tickers(
    sqlite_engine: Engine,
    universe_config_dir: Path,
) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(ticker="AAK.ST", trading_dates=[date(2026, 6, 1)]),
                _daily_prices(ticker="BOL.ST", trading_dates=[date(2026, 6, 10)]),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )

    result = check_active_ticker_bronze_onboarding(
        config_dir=universe_config_dir,
        engine=sqlite_engine,
    )

    states = {item.ticker: item for item in result.states}
    assert result.tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert states["AAK.ST"].status == BronzeOnboardingStatus.ONBOARDED
    assert states["AAK.ST"].row_count == 1
    assert states["AAK.ST"].first_trading_date == date(2026, 6, 1)
    assert states["AAK.ST"].last_trading_date == date(2026, 6, 1)
    assert states["BOL.ST"].status == BronzeOnboardingStatus.ONBOARDED
    assert states["VOLV-B.ST"].status == BronzeOnboardingStatus.MISSING
    assert states["VOLV-B.ST"].row_count == 0
    assert result.onboarded_count == 2
    assert result.missing_count == 1
    assert result.missing_tickers == ("VOLV-B.ST",)


def test_check_active_ticker_bronze_onboarding_treats_short_history_as_onboarded(
    sqlite_engine: Engine,
    universe_config_dir: Path,
) -> None:
    upsert_daily_prices(
        prices=_daily_prices(ticker="AAK.ST", trading_dates=[date(2026, 6, 10)]),
        engine=sqlite_engine,
    )

    result = check_active_ticker_bronze_onboarding(
        config_dir=universe_config_dir,
        engine=sqlite_engine,
    )

    states = {item.ticker: item for item in result.states}
    assert states["AAK.ST"].status == BronzeOnboardingStatus.ONBOARDED
    assert result.missing_tickers == ("BOL.ST", "VOLV-B.ST")


def test_check_active_ticker_bronze_onboarding_initializes_empty_database(
    sqlite_engine: Engine,
    universe_config_dir: Path,
) -> None:
    result = check_active_ticker_bronze_onboarding(
        config_dir=universe_config_dir,
        engine=sqlite_engine,
    )

    assert result.missing_count == 3
    assert result.onboarded_count == 0
    assert result.missing_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")


def test_check_active_ticker_bronze_onboarding_limits_resolved_active_tickers(
    sqlite_engine: Engine,
    universe_config_dir: Path,
) -> None:
    result = check_active_ticker_bronze_onboarding(
        config_dir=universe_config_dir,
        limit=2,
        engine=sqlite_engine,
    )

    assert result.tickers == ("AAK.ST", "BOL.ST")
    assert [item.ticker for item in result.states] == ["AAK.ST", "BOL.ST"]


def test_check_active_ticker_bronze_onboarding_validates_limit(
    sqlite_engine: Engine,
    universe_config_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="limit must be greater than zero"):
        check_active_ticker_bronze_onboarding(
            config_dir=universe_config_dir,
            limit=0,
            engine=sqlite_engine,
        )


def test_sync_active_ticker_bronze_onboarding_does_not_backfill_by_default(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(**kwargs: object) -> IngestionResult:
        raise AssertionError("backfill should not run by default")

    monkeypatch.setattr(onboarding, "ingest_historical_daily_prices", fail_if_called)

    result = sync_active_ticker_bronze_onboarding(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        config_dir=universe_config_dir,
        engine=sqlite_engine,
    )

    assert result.backfill_tickers == ()
    assert result.ingestion_result is None
    assert result.onboarding_before.missing_count == 3
    assert result.onboarding_after is None


def test_sync_active_ticker_bronze_onboarding_backfills_only_missing_tickers(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(ticker="AAK.ST", trading_dates=[date(2026, 6, 1)]),
                _daily_prices(ticker="BOL.ST", trading_dates=[date(2026, 6, 10)]),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )
    calls: list[dict[str, object]] = []

    def fake_ingest_historical_daily_prices(**kwargs: object) -> IngestionResult:
        tickers = cast(tuple[str, ...], kwargs["tickers"])
        start = cast(date, kwargs["start_date"])
        end = cast(date, kwargs["end_date"])
        calls.append(kwargs)
        return IngestionResult(
            provider="yfinance",
            request_id="backfill-request",
            start_date=start,
            end_date=end,
            tickers=tickers,
            downloaded_rows=1,
            upserted_rows=1,
            failures=(),
        )

    monkeypatch.setattr(
        onboarding,
        "ingest_historical_daily_prices",
        fake_ingest_historical_daily_prices,
    )

    result = sync_active_ticker_bronze_onboarding(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        config_dir=universe_config_dir,
        engine=sqlite_engine,
        backfill=True,
        raise_on_failure=True,
    )

    assert result.backfill_tickers == ("VOLV-B.ST",)
    assert result.ingestion_result is not None
    assert result.ingestion_result.tickers == ("VOLV-B.ST",)
    assert len(calls) == 1
    assert calls[0]["tickers"] == ("VOLV-B.ST",)
    assert calls[0]["start_date"] == date(2026, 6, 1)
    assert calls[0]["end_date"] == date(2026, 6, 30)
    assert calls[0]["engine"] is sqlite_engine
    assert calls[0]["raise_on_failure"] is True


def test_sync_active_ticker_bronze_onboarding_reports_post_backfill_state(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ingest_historical_daily_prices(**kwargs: object) -> IngestionResult:
        tickers = cast(tuple[str, ...], kwargs["tickers"])
        start = cast(date, kwargs["start_date"])
        end = cast(date, kwargs["end_date"])
        upsert_daily_prices(
            prices=pd.concat(
                [
                    _daily_prices(ticker=ticker, trading_dates=[date(2026, 6, 1)])
                    for ticker in tickers
                ],
                ignore_index=True,
            ),
            engine=sqlite_engine,
        )
        return IngestionResult(
            provider="yfinance",
            request_id="backfill-request",
            start_date=start,
            end_date=end,
            tickers=tickers,
            downloaded_rows=len(tickers),
            upserted_rows=len(tickers),
            failures=(),
        )

    monkeypatch.setattr(
        onboarding,
        "ingest_historical_daily_prices",
        fake_ingest_historical_daily_prices,
    )

    result = sync_active_ticker_bronze_onboarding(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        config_dir=universe_config_dir,
        engine=sqlite_engine,
        backfill=True,
    )

    assert result.onboarding_before.missing_count == 3
    assert result.onboarding_after is not None
    assert result.onboarding_after.missing_count == 0
    assert result.onboarding_after.onboarded_count == 3


def test_sync_active_ticker_bronze_onboarding_skips_backfill_when_everything_is_onboarded(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(ticker="AAK.ST", trading_dates=[date(2026, 6, 1)]),
                _daily_prices(ticker="BOL.ST", trading_dates=[date(2026, 6, 10)]),
                _daily_prices(ticker="VOLV-B.ST", trading_dates=[date(2026, 6, 29)]),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )

    def fail_if_called(**kwargs: object) -> IngestionResult:
        raise AssertionError("onboarded tickers should not be backfilled")

    monkeypatch.setattr(onboarding, "ingest_historical_daily_prices", fail_if_called)

    result = sync_active_ticker_bronze_onboarding(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        config_dir=universe_config_dir,
        engine=sqlite_engine,
        backfill=True,
    )

    assert result.onboarding_before.onboarded_count == 3
    assert result.backfill_tickers == ()
    assert result.ingestion_result is None
    assert result.onboarding_after is None


def test_sync_active_ticker_bronze_onboarding_reruns_do_not_duplicate_bronze_rows(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_ingest_historical_daily_prices(**kwargs: object) -> IngestionResult:
        tickers = cast(tuple[str, ...], kwargs["tickers"])
        start = cast(date, kwargs["start_date"])
        end = cast(date, kwargs["end_date"])
        calls.append(tickers)
        upsert_daily_prices(
            prices=pd.concat(
                [
                    _daily_prices(ticker=ticker, trading_dates=[date(2026, 6, 1)])
                    for ticker in tickers
                ],
                ignore_index=True,
            ),
            engine=sqlite_engine,
        )
        return IngestionResult(
            provider="yfinance",
            request_id="backfill-request",
            start_date=start,
            end_date=end,
            tickers=tickers,
            downloaded_rows=len(tickers),
            upserted_rows=len(tickers),
            failures=(),
        )

    monkeypatch.setattr(
        onboarding,
        "ingest_historical_daily_prices",
        fake_ingest_historical_daily_prices,
    )

    for _ in range(2):
        sync_active_ticker_bronze_onboarding(
            start_date=date(2026, 6, 1),
            end_date=date(2026, 6, 30),
            config_dir=universe_config_dir,
            engine=sqlite_engine,
            backfill=True,
        )

    with sqlite_engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))

    assert calls == [("AAK.ST", "BOL.ST", "VOLV-B.ST")]
    assert row_count == 3


def test_sync_active_ticker_bronze_onboarding_validates_date_window(
    sqlite_engine: Engine,
    universe_config_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="start_date must be before end_date"):
        sync_active_ticker_bronze_onboarding(
            start_date=date(2026, 6, 30),
            end_date=date(2026, 6, 30),
            config_dir=universe_config_dir,
            engine=sqlite_engine,
        )


def _write_config(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _daily_prices(*, ticker: str, trading_dates: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": "yfinance",
                "ticker": ticker,
                "trading_date": trading_date,
                "open": 275.4,
                "high": 279.0,
                "low": 274.8,
                "close": 278.2,
                "adjusted_close": None,
                "volume": 124500,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "fetched_at": datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
                "request_id": "test-request",
            }
            for trading_date in trading_dates
        ],
        columns=DAILY_PRICE_COLUMNS,
    )
