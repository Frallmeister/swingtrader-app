from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from swingtrader.data.bronze.schema import metadata
from swingtrader.data.bronze.writer import upsert_daily_prices
from swingtrader.data.clients.yfinance import DAILY_PRICE_COLUMNS
from swingtrader.data.eligibility import (
    EligibilityFailureReason,
    InferenceReadinessStatus,
    TrainingEligibilityStatus,
    check_inference_readiness,
    check_training_eligibility,
    get_inference_ready_tickers,
    get_training_eligible_tickers,
)


@pytest.fixture
def sqlite_engine() -> Engine:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    return engine


def test_check_inference_readiness_marks_ready_ticker(sqlite_engine: Engine) -> None:
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="READY.ST",
            row_count=252,
            last_trading_date=date(2026, 7, 3),
        ),
        engine=sqlite_engine,
    )

    result = check_inference_readiness(
        tickers=("READY.ST",),
        reference_date=date(2026, 7, 4),
        engine=sqlite_engine,
    )

    assert result.ready_tickers == ("READY.ST",)
    state = result.states[0]
    assert state.status == InferenceReadinessStatus.READY
    assert state.failure_reasons == ()
    assert state.row_count == 252
    assert state.days_since_latest_row == 1
    assert state.latest_median_turnover == Decimal("10000000.000000")


def test_check_inference_readiness_reports_bronze_blockers(sqlite_engine: Engine) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(
                    ticker="SHORT.ST",
                    row_count=251,
                    last_trading_date=date(2026, 7, 3),
                ),
                _daily_prices(
                    ticker="STALE.ST",
                    row_count=252,
                    last_trading_date=date(2026, 7, 2),
                ),
                _daily_prices(
                    ticker="NOADJ.ST",
                    row_count=252,
                    last_trading_date=date(2026, 7, 3),
                    missing_adjusted_close_indexes={0},
                ),
                _daily_prices(
                    ticker="SPARSE.ST",
                    row_count=252,
                    last_trading_date=date(2026, 7, 3),
                    zero_volume_indexes=set(range(14)),
                ),
                _daily_prices(
                    ticker="ILLIQUID.ST",
                    row_count=252,
                    last_trading_date=date(2026, 7, 3),
                    volume=10_000,
                ),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )

    result = check_inference_readiness(
        tickers=(
            "MISSING.ST",
            "SHORT.ST",
            "STALE.ST",
            "NOADJ.ST",
            "SPARSE.ST",
            "ILLIQUID.ST",
        ),
        reference_date=date(2026, 7, 4),
        engine=sqlite_engine,
    )

    states = {state.ticker: state for state in result.states}
    assert result.not_ready_tickers == (
        "MISSING.ST",
        "SHORT.ST",
        "STALE.ST",
        "NOADJ.ST",
        "SPARSE.ST",
        "ILLIQUID.ST",
    )
    assert states["MISSING.ST"].failure_reasons == (
        EligibilityFailureReason.MISSING_BRONZE_DAILY_PRICES,
    )
    assert states["SHORT.ST"].failure_reasons == (EligibilityFailureReason.INSUFFICIENT_HISTORY,)
    assert states["STALE.ST"].failure_reasons == (EligibilityFailureReason.STALE_DAILY_PRICES,)
    assert states["NOADJ.ST"].failure_reasons == (EligibilityFailureReason.MISSING_ADJUSTED_CLOSE,)
    assert states["SPARSE.ST"].failure_reasons == (EligibilityFailureReason.SPARSE_VOLUME,)
    assert states["ILLIQUID.ST"].failure_reasons == (EligibilityFailureReason.LOW_LIQUIDITY,)


def test_check_training_eligibility_accepts_explicit_broader_tickers(
    sqlite_engine: Engine,
) -> None:
    upsert_daily_prices(
        prices=_daily_prices(
            ticker="BROAD.ST",
            row_count=756,
            last_trading_date=date(2026, 7, 3),
        ),
        engine=sqlite_engine,
    )

    result = check_training_eligibility(tickers=("BROAD.ST",), engine=sqlite_engine)

    assert result.eligible_tickers == ("BROAD.ST",)
    state = result.states[0]
    assert state.status == TrainingEligibilityStatus.ELIGIBLE
    assert state.failure_reasons == ()
    assert state.row_count == 756


def test_check_training_eligibility_reports_history_and_quality_blockers(
    sqlite_engine: Engine,
) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(
                    ticker="SHORT.ST",
                    row_count=755,
                    last_trading_date=date(2026, 7, 3),
                ),
                _daily_prices(
                    ticker="NOADJ.ST",
                    row_count=756,
                    last_trading_date=date(2026, 7, 3),
                    missing_adjusted_close_indexes={0},
                ),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(
                    ticker="SPARSE.ST",
                    row_count=756,
                    last_trading_date=date(2026, 7, 3),
                    zero_volume_indexes=set(range(39)),
                ),
                _daily_prices(
                    ticker="ILLIQUID.ST",
                    row_count=756,
                    last_trading_date=date(2026, 7, 3),
                    volume=10_000,
                ),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )

    result = check_training_eligibility(
        tickers=("SHORT.ST", "NOADJ.ST", "SPARSE.ST", "ILLIQUID.ST"),
        engine=sqlite_engine,
    )

    states = {state.ticker: state for state in result.states}
    assert result.not_eligible_tickers == (
        "SHORT.ST",
        "NOADJ.ST",
        "SPARSE.ST",
        "ILLIQUID.ST",
    )
    assert states["SHORT.ST"].failure_reasons == (EligibilityFailureReason.INSUFFICIENT_HISTORY,)
    assert states["NOADJ.ST"].failure_reasons == (EligibilityFailureReason.MISSING_ADJUSTED_CLOSE,)
    assert states["SPARSE.ST"].failure_reasons == (EligibilityFailureReason.SPARSE_VOLUME,)
    assert states["ILLIQUID.ST"].failure_reasons == (EligibilityFailureReason.LOW_LIQUIDITY,)


def test_eligibility_helpers_return_only_passing_tickers(sqlite_engine: Engine) -> None:
    upsert_daily_prices(
        prices=pd.concat(
            [
                _daily_prices(
                    ticker="READY.ST",
                    row_count=756,
                    last_trading_date=date(2026, 7, 3),
                ),
                _daily_prices(
                    ticker="SHORT.ST",
                    row_count=251,
                    last_trading_date=date(2026, 7, 3),
                ),
            ],
            ignore_index=True,
        ),
        engine=sqlite_engine,
    )

    assert get_inference_ready_tickers(
        tickers=("READY.ST", "SHORT.ST"),
        reference_date=date(2026, 7, 4),
        engine=sqlite_engine,
    ) == ("READY.ST",)
    assert get_training_eligible_tickers(
        tickers=("READY.ST", "SHORT.ST"),
        engine=sqlite_engine,
    ) == ("READY.ST",)


def _daily_prices(
    *,
    ticker: str,
    row_count: int,
    last_trading_date: date,
    close: Decimal = Decimal("100.00"),
    adjusted_close: Decimal = Decimal("100.00"),
    volume: int = 100_000,
    missing_adjusted_close_indexes: set[int] | None = None,
    zero_volume_indexes: set[int] | None = None,
) -> pd.DataFrame:
    missing_adjusted_close_indexes = missing_adjusted_close_indexes or set()
    zero_volume_indexes = zero_volume_indexes or set()
    first_trading_date = last_trading_date - timedelta(days=row_count - 1)
    return pd.DataFrame(
        [
            {
                "provider": "yfinance",
                "ticker": ticker,
                "trading_date": first_trading_date + timedelta(days=index),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adjusted_close": (
                    None if index in missing_adjusted_close_indexes else adjusted_close
                ),
                "volume": 0 if index in zero_volume_indexes else volume,
                "dividends": Decimal("0.00"),
                "stock_splits": Decimal("0.00"),
                "fetched_at": datetime(2026, 7, 4, 9, 30, tzinfo=UTC),
                "request_id": "test-request",
            }
            for index in range(row_count)
        ],
        columns=DAILY_PRICE_COLUMNS,
    )
