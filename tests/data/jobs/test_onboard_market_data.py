from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.schema import bronze_market_daily_prices, metadata
from swingtrader.data.bronze.writer import upsert_daily_prices
from swingtrader.data.clients.yfinance import DAILY_PRICE_COLUMNS
from swingtrader.data.ingestion import market_data
from swingtrader.data.ingestion.market_data import IngestionResult, TickerIngestionFailure
from swingtrader.data.jobs import onboard_market_data


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
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


def test_run_onboard_market_data_fills_empty_bronze_for_active_tickers(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = onboard_market_data.run_onboard_market_data(
        end_date=date(2000, 1, 3),
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == [
        ("AAK.ST", date(2000, 1, 1), date(2000, 1, 3)),
        ("BOL.ST", date(2000, 1, 1), date(2000, 1, 3)),
        ("VOLV-B.ST", date(2000, 1, 1), date(2000, 1, 3)),
    ]
    assert result.active_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.already_onboarded_tickers == ()
    assert result.missing_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.attempted_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.successful_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.failed_tickers == ()
    assert result.downloaded_rows == 3
    assert result.upserted_rows == 3
    assert result.onboarding_after is not None
    assert result.onboarding_after.missing_count == 0
    assert _stored_row_count(sqlite_engine) == 3


def test_run_onboard_market_data_downloads_only_missing_active_tickers(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="AAK.ST",
            trading_date=date(2026, 6, 1),
            fetched_at=datetime(2026, 6, 1, 9, 30, tzinfo=UTC),
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

    result = onboard_market_data.run_onboard_market_data(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 7, 4),
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == ["BOL.ST", "VOLV-B.ST"]
    assert result.already_onboarded_tickers == ("AAK.ST",)
    assert result.missing_tickers == ("BOL.ST", "VOLV-B.ST")
    assert result.attempted_tickers == ("BOL.ST", "VOLV-B.ST")
    assert result.downloaded_rows == 2
    assert _stored_row_count(sqlite_engine) == 3


def test_run_onboard_market_data_is_noop_when_all_active_tickers_are_onboarded(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(
                    ticker="AAK.ST",
                    trading_date=date(2026, 6, 1),
                    fetched_at=datetime(2026, 6, 1, 9, 30, tzinfo=UTC),
                    request_id="existing-request",
                ),
                _daily_prices(
                    ticker="BOL.ST",
                    trading_date=date(2026, 6, 10),
                    fetched_at=datetime(2026, 6, 10, 9, 30, tzinfo=UTC),
                    request_id="existing-request",
                ),
                _daily_prices(
                    ticker="VOLV-B.ST",
                    trading_date=date(2026, 6, 29),
                    fetched_at=datetime(2026, 6, 29, 9, 30, tzinfo=UTC),
                    request_id="existing-request",
                ),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )

    def fail_if_called(**kwargs: object) -> pd.DataFrame:
        raise AssertionError("onboarded tickers should not be downloaded")

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fail_if_called,
    )

    with caplog.at_level("INFO", logger=onboard_market_data.__name__):
        result = onboard_market_data.run_onboard_market_data(
            end_date=date(2026, 7, 4),
            engine=sqlite_engine,
            config_dir=universe_config_dir,
            settings_path=market_data_settings_path,
        )

    assert result.already_onboarded_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.missing_tickers == ()
    assert result.attempted_tickers == ()
    assert result.ingestion_result is None
    assert result.onboarding_after is None
    assert "all active tickers are already onboarded" in caplog.text


def test_run_onboard_market_data_applies_limit_before_onboarding(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    result = onboard_market_data.run_onboard_market_data(
        end_date=date(2000, 1, 3),
        limit=2,
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == ["AAK.ST", "BOL.ST"]
    assert result.active_tickers == ("AAK.ST", "BOL.ST")
    assert result.attempted_tickers == ("AAK.ST", "BOL.ST")


def test_run_onboard_market_data_uses_configured_initial_start_date_by_default(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[date] = []

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        calls.append(_date_arg(kwargs, "start_date"))
        return _daily_prices(
            ticker=_single_ticker(kwargs),
            trading_date=_date_arg(kwargs, "start_date"),
            fetched_at=kwargs["fetched_at"],
            request_id=kwargs["request_id"],
        )

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = onboard_market_data.run_onboard_market_data(
        end_date=date(2000, 1, 3),
        limit=1,
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == [date(2000, 1, 1)]
    assert result.start_date == date(2000, 1, 1)


def test_run_onboard_market_data_start_and_end_date_arguments_override_defaults(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[date, date]] = []

    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        start_date = _date_arg(kwargs, "start_date")
        end_date = _date_arg(kwargs, "end_date")
        calls.append((start_date, end_date))
        return _daily_prices(
            ticker=_single_ticker(kwargs),
            trading_date=start_date,
            fetched_at=kwargs["fetched_at"],
            request_id=kwargs["request_id"],
        )

    monkeypatch.setattr(
        market_data.yfinance_client,
        "download_daily_prices",
        fake_download_daily_prices,
    )

    result = onboard_market_data.run_onboard_market_data(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        limit=1,
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert calls == [(date(2024, 1, 1), date(2024, 2, 1))]
    assert result.start_date == date(2024, 1, 1)
    assert result.end_date == date(2024, 2, 1)


def test_run_onboard_market_data_reports_failed_tickers_without_discarding_successes(
    sqlite_engine: Engine,
    universe_config_dir: Path,
    market_data_settings_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download_daily_prices(**kwargs: object) -> pd.DataFrame:
        ticker = _single_ticker(kwargs)
        if ticker == "BOL.ST":
            raise RuntimeError("download failed")
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

    result = onboard_market_data.run_onboard_market_data(
        end_date=date(2000, 1, 3),
        engine=sqlite_engine,
        config_dir=universe_config_dir,
        settings_path=market_data_settings_path,
    )

    assert result.attempted_tickers == ("AAK.ST", "BOL.ST", "VOLV-B.ST")
    assert result.successful_tickers == ("AAK.ST", "VOLV-B.ST")
    assert result.failed_tickers == ("BOL.ST",)
    assert result.downloaded_rows == 2
    assert result.upserted_rows == 2
    assert _stored_row_count(sqlite_engine) == 2


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
    onboard_result = onboard_market_data.OnboardMarketDataResult(
        provider="yfinance",
        start_date=date(2026, 7, 1),
        end_date=date(2026, 7, 4),
        active_tickers=("FAIL.ST",),
        already_onboarded_tickers=(),
        missing_tickers=("FAIL.ST",),
        attempted_tickers=("FAIL.ST",),
        ingestion_result=ingestion_result,
        onboarding_after=None,
    )

    monkeypatch.setattr(
        onboard_market_data,
        "run_onboard_market_data",
        lambda **_: onboard_result,
    )

    assert onboard_market_data.main(["--fail-on-ticker-failure"]) == 1
    assert onboard_market_data.main([]) == 0


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
