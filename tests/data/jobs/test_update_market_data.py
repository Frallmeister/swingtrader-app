from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.queries import BronzeDailyPriceState
from swingtrader.data.bronze.schema import bronze_market_daily_prices
from swingtrader.data.bronze.writer import upsert_daily_prices
from swingtrader.data.clients.yfinance import DAILY_PRICE_COLUMNS
from swingtrader.data.db import initialize_database
from swingtrader.data.ingestion import market_data
from swingtrader.data.ingestion.market_data import IngestionResult, TickerIngestionFailure
from swingtrader.data.jobs import update_market_data


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
  - ticker: AAK.ST
    asset_type: EQUITY
  - ticker: BOL.ST
    asset_type: EQUITY
  - ticker: VOLV-B.ST
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


@pytest.fixture
def market_data_settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "market_data.yml"
    _write_config(
        path,
        """
kind: market_data_settings
provider: yfinance
initial_start_date: 2000-01-01
""".lstrip(),
    )
    return path


def test_plan_daily_market_data_updates_returns_one_update_per_onboarded_ticker() -> None:
    planned_updates = update_market_data.plan_daily_market_data_updates(
        active_tickers=("AAK.ST", "BOL.ST", "VOLV-B.ST"),
        state_by_ticker={
            "AAK.ST": BronzeDailyPriceState(
                ticker="AAK.ST",
                row_count=10,
                first_trading_date=date(2026, 6, 1),
                last_trading_date=date(2026, 6, 26),
            ),
            "BOL.ST": BronzeDailyPriceState(
                ticker="BOL.ST",
                row_count=12,
                first_trading_date=date(2026, 6, 1),
                last_trading_date=date(2026, 6, 26),
            ),
        },
        end_date=date(2026, 7, 4),
    )

    assert planned_updates == (
        update_market_data.DailyMarketDataPlannedUpdate(
            start_date=date(2026, 6, 26),
            end_date=date(2026, 7, 4),
            ticker="AAK.ST",
        ),
        update_market_data.DailyMarketDataPlannedUpdate(
            start_date=date(2026, 6, 26),
            end_date=date(2026, 7, 4),
            ticker="BOL.ST",
        ),
    )


def test_run_daily_market_data_update_reports_empty_bronze_as_not_onboarded(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[tuple[str, date, date]] = []

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        raise AssertionError("daily update should not initialize missing tickers")

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    with caplog.at_level("WARNING", logger=update_market_data.__name__):
        result = update_market_data.run_daily_market_data_update(
            end_date=date(2000, 1, 3),
            engine=sqlite_engine,
            config_dir=universe_config_dir,
            settings_path=market_data_settings_path,
        )

    assert calls == []
    assert result.not_onboarded_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.skipped_tickers == ()
    assert result.downloaded_rows == 0
    assert result.upserted_rows == 0
    assert result.failures == ()
    assert _stored_row_count(sqlite_engine) == 0
    assert "no planned updates because no active tickers are onboarded" in caplog.text


def test_run_daily_market_data_update_skips_current_onboarded_tickers(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="AAK.ST",
            trading_date=date(2026, 7, 4),
            fetched_at=datetime(2026, 7, 4, 9, 30, tzinfo=UTC),
            request_id="existing-request",
        ),
        engine=sqlite_engine,
    )

    def fail_if_called(**kwargs: object) -> pd.DataFrame:
        raise AssertionError("current onboarded tickers should be skipped")

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fail_if_called,
    )

    result = update_market_data.run_daily_market_data_update(
        end_date=date(2026, 7, 4),
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert result.update_tickers == ()
    assert result.skipped_tickers == ("AAK.ST",)
    assert result.not_onboarded_tickers == ("BOL.ST", "VOLV-B.ST")


def test_run_daily_market_data_update_uses_latest_ticker_dates(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="AAK.ST",
            trading_date=date(2026, 6, 26),
            fetched_at=datetime(2026, 6, 26, 9, 30, tzinfo=UTC),
            request_id="existing-request",
        ),
        engine=sqlite_engine,
    )
    calls: list[tuple[str, date, date]] = []

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        ticker = _single_ticker(kwargs)
        start_date = _date_arg(kwargs, "start_date")
        end_date = _date_arg(kwargs, "end_date")
        calls.append((ticker, start_date, end_date))
        return _daily_prices(
            ticker=ticker,
            trading_date=start_date,
            fetched_at=kwargs["fetched_at"],
            request_id=kwargs["request_id"],
        )

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = update_market_data.run_daily_market_data_update(
        end_date=date(2026, 7, 4),
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == [
        ("AAK.ST", date(2026, 6, 26), date(2026, 7, 4)),
    ]
    assert result.update_tickers == ("AAK.ST",)
    assert result.not_onboarded_tickers == ("BOL.ST", "VOLV-B.ST")
    assert result.skipped_tickers == ()
    assert _stored_row_count(sqlite_engine) == 1


def test_run_daily_market_data_update_applies_limit(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="AAK.ST",
            trading_date=date(2000, 1, 1),
            fetched_at=datetime(2000, 1, 1, 9, 30, tzinfo=UTC),
            request_id="existing-request",
        ),
        engine=sqlite_engine,
    )
    calls: list[str] = []

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        ticker = _single_ticker(kwargs)
        calls.append(ticker)
        return _daily_prices(
            ticker=ticker,
            trading_date=_date_arg(kwargs, "start_date"),
            fetched_at=kwargs["fetched_at"],
            request_id=kwargs["request_id"],
        )

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = update_market_data.run_daily_market_data_update(
        end_date=date(2000, 1, 3),
        limit=1,
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == ["AAK.ST"]
    assert result.active_tickers == ("AAK.ST",)
    assert result.not_onboarded_tickers == ()


def test_main_returns_nonzero_when_fail_on_ticker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = TickerIngestionFailure(
        ticker="FAIL.ST",
        error_type="RuntimeError",
        message="download failed",
    )
    ingestion_result = IngestionResult(
        provider="yfinance",
        request_id="test-request",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 4),
        tickers=("FAIL.ST",),
        downloaded_rows=0,
        upserted_rows=0,
        failures=(failure,),
    )
    update_result = update_market_data.DailyMarketDataUpdateResult(
        provider="yfinance",
        end_date=date(2026, 7, 4),
        active_tickers=("FAIL.ST",),
        not_onboarded_tickers=(),
        skipped_tickers=(),
        planned_updates=(
            update_market_data.DailyMarketDataPlannedUpdate(
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 4),
                ticker="FAIL.ST",
            ),
        ),
        ingestion_results=(ingestion_result,),
    )

    monkeypatch.setattr(
        update_market_data,
        "run_daily_market_data_update",
        lambda **_: update_result,
    )

    assert update_market_data.main(["--fail-on-ticker-failure"]) == 1
    assert update_market_data.main([]) == 0


def _write_config(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _single_ticker(kwargs: dict[str, object]) -> str:
    tickers = kwargs["tickers"]
    assert isinstance(tickers, list)
    assert len(tickers) == 1
    return str(tickers[0])


def _date_arg(kwargs: dict[str, object], name: str) -> date:
    value = kwargs[name]
    assert isinstance(value, date)
    return value


def _daily_prices(
    *,
    ticker: str,
    trading_date: date,
    fetched_at: object,
    request_id: object,
) -> pd.DataFrame:
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
                "fetched_at": fetched_at,
                "request_id": request_id,
            }
        ],
        columns=DAILY_PRICE_COLUMNS,
    )


def _stored_row_count(engine: Engine) -> int:
    with engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(bronze_market_daily_prices))
    assert row_count is not None
    return row_count
