"""Ticker inference readiness and training eligibility checks.

Active ticker configuration is desired production/trading state. This module evaluates actual
stored data state for model inference and training. The first implementation is bronze-backed;
feature and label table checks should become additional hard blockers once those tables exist.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from math import ceil

from sqlalchemy.engine import Engine

from swingtrader.core.db import resolve_database_engine
from swingtrader.data.bronze.queries import (
    BronzeDailyPriceQualityState,
    BronzeDailyPriceState,
    load_daily_price_quality_state_by_ticker,
    load_daily_price_state_by_ticker,
)
from swingtrader.data.clients import yfinance as yfinance_client
from swingtrader.data.ingestion.universe_selection import ConfigDir, resolve_requested_tickers

INFERENCE_READINESS_MIN_DAILY_PRICE_ROWS = 252
INFERENCE_READINESS_MAX_DAYS_SINCE_LATEST = 4
TRAINING_ELIGIBILITY_MIN_DAILY_PRICE_ROWS = 756
QUALITY_MAX_NULL_OR_ZERO_VOLUME_RATIO = Decimal("0.05")
QUALITY_TURNOVER_LOOKBACK_ROWS = 60
# TODO: Convert foreign-market prices to SEK before applying this threshold to non-SEK tickers.
QUALITY_MIN_MEDIAN_TURNOVER = Decimal("5000000")
QUALITY_MIN_TURNOVER_OBSERVATION_ROWS = ceil(
    QUALITY_TURNOVER_LOOKBACK_ROWS * (1 - float(QUALITY_MAX_NULL_OR_ZERO_VOLUME_RATIO))
)


class InferenceReadinessStatus(StrEnum):
    """Inference readiness status for one ticker."""

    READY = "ready"
    NOT_READY = "not_ready"


class TrainingEligibilityStatus(StrEnum):
    """Training eligibility status for one ticker."""

    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"


class EligibilityFailureReason(StrEnum):
    """Hard blocker reasons for inference readiness or training eligibility."""

    MISSING_BRONZE_DAILY_PRICES = "missing_bronze_daily_prices"
    INSUFFICIENT_HISTORY = "insufficient_history"
    STALE_DAILY_PRICES = "stale_daily_prices"
    MISSING_ADJUSTED_CLOSE = "missing_adjusted_close"
    SPARSE_VOLUME = "sparse_volume"
    INSUFFICIENT_TURNOVER_OBSERVATIONS = "insufficient_turnover_observations"
    LOW_LIQUIDITY = "low_liquidity"


@dataclass(frozen=True)
class TickerInferenceReadinessState:
    """Inference readiness summary for one ticker."""

    ticker: str
    status: InferenceReadinessStatus
    failure_reasons: tuple[EligibilityFailureReason, ...]
    row_count: int
    first_trading_date: date | None
    last_trading_date: date | None
    days_since_latest_row: int | None
    missing_adjusted_close_count: int
    null_or_zero_volume_count: int
    null_or_zero_volume_ratio: Decimal | None
    latest_turnover_row_count: int
    latest_median_turnover: Decimal | None


@dataclass(frozen=True)
class InferenceReadinessResult:
    """Summary of inference readiness for requested tickers."""

    provider: str
    reference_date: date
    tickers: tuple[str, ...]
    states: tuple[TickerInferenceReadinessState, ...]

    @property
    def ready_tickers(self) -> tuple[str, ...]:
        return tuple(
            state.ticker for state in self.states if state.status == InferenceReadinessStatus.READY
        )

    @property
    def not_ready_tickers(self) -> tuple[str, ...]:
        return tuple(
            state.ticker
            for state in self.states
            if state.status == InferenceReadinessStatus.NOT_READY
        )


@dataclass(frozen=True)
class TickerTrainingEligibilityState:
    """Training eligibility summary for one ticker."""

    ticker: str
    status: TrainingEligibilityStatus
    failure_reasons: tuple[EligibilityFailureReason, ...]
    row_count: int
    first_trading_date: date | None
    last_trading_date: date | None
    missing_adjusted_close_count: int
    null_or_zero_volume_count: int
    null_or_zero_volume_ratio: Decimal | None
    latest_turnover_row_count: int
    latest_median_turnover: Decimal | None


@dataclass(frozen=True)
class TrainingEligibilityResult:
    """Summary of training eligibility for requested tickers."""

    provider: str
    tickers: tuple[str, ...]
    states: tuple[TickerTrainingEligibilityState, ...]

    @property
    def eligible_tickers(self) -> tuple[str, ...]:
        return tuple(
            state.ticker
            for state in self.states
            if state.status == TrainingEligibilityStatus.ELIGIBLE
        )

    @property
    def not_eligible_tickers(self) -> tuple[str, ...]:
        return tuple(
            state.ticker
            for state in self.states
            if state.status == TrainingEligibilityStatus.NOT_ELIGIBLE
        )


def check_inference_readiness(
    *,
    provider: str = yfinance_client.PROVIDER,
    reference_date: date | None = None,
    tickers: Sequence[str] | None = None,
    limit: int | None = None,
    config_dir: ConfigDir | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
) -> InferenceReadinessResult:
    """Check whether requested tickers are ready for production inference.

    When ``tickers`` is omitted, the active trading universe is resolved from configuration.
    Passing explicit tickers lets future callers evaluate a broader universe without treating
    active status as a readiness rule.

    Parameters
    ----------
    provider
        Market data provider to evaluate in bronze state.
    reference_date
        Date used to decide whether the latest bronze row is stale. When omitted, today's UTC
        date is used.
    tickers
        Optional explicit ticker symbols to evaluate. When omitted, active tickers are
        resolved from configuration.
    limit
        Optional maximum number of normalized tickers to evaluate.
    config_dir
        Optional active ticker configuration directory used when ``tickers`` is omitted.
    database_url
        Optional SQLAlchemy database URL. Mutually exclusive with ``engine``.
    engine
        Optional SQLAlchemy engine. Passing an engine is useful for tests and callers that
        already manage database connections. Mutually exclusive with ``database_url``.

    Returns
    -------
    InferenceReadinessResult
        Readiness result with one state per requested ticker.
    """
    resolved_reference_date = reference_date or datetime.now(UTC).date()
    resolved_tickers = resolve_requested_tickers(
        tickers=tickers,
        limit=limit,
        config_dir=config_dir,
    )
    resolved_engine = resolve_database_engine(database_url=database_url, engine=engine)
    state_by_ticker = load_daily_price_state_by_ticker(
        engine=resolved_engine,
        provider=provider,
        tickers=resolved_tickers,
    )
    quality_by_ticker = load_daily_price_quality_state_by_ticker(
        engine=resolved_engine,
        provider=provider,
        tickers=resolved_tickers,
        turnover_lookback_rows=QUALITY_TURNOVER_LOOKBACK_ROWS,
    )
    return InferenceReadinessResult(
        provider=provider,
        reference_date=resolved_reference_date,
        tickers=resolved_tickers,
        states=tuple(
            _build_inference_readiness_state(
                ticker=ticker,
                stored_state=state_by_ticker.get(ticker),
                quality_state=quality_by_ticker.get(ticker),
                reference_date=resolved_reference_date,
            )
            for ticker in resolved_tickers
        ),
    )


def check_training_eligibility(
    *,
    provider: str = yfinance_client.PROVIDER,
    tickers: Sequence[str] | None = None,
    limit: int | None = None,
    config_dir: ConfigDir | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
) -> TrainingEligibilityResult:
    """Check whether requested tickers are eligible for model training.

    This first version uses bronze history and quality gates only. Future feature and label
    tables should add feature-history and label-count gates before model training consumes
    these results.

    Parameters
    ----------
    provider
        Market data provider to evaluate in bronze state.
    tickers
        Optional explicit ticker symbols to evaluate. When omitted, active tickers are
        resolved from configuration.
    limit
        Optional maximum number of normalized tickers to evaluate.
    config_dir
        Optional active ticker configuration directory used when ``tickers`` is omitted.
    database_url
        Optional SQLAlchemy database URL. Mutually exclusive with ``engine``.
    engine
        Optional SQLAlchemy engine. Passing an engine is useful for tests and callers that
        already manage database connections. Mutually exclusive with ``database_url``.

    Returns
    -------
    TrainingEligibilityResult
        Eligibility result with one state per requested ticker.
    """
    resolved_tickers = resolve_requested_tickers(
        tickers=tickers,
        limit=limit,
        config_dir=config_dir,
    )
    resolved_engine = resolve_database_engine(database_url=database_url, engine=engine)
    state_by_ticker = load_daily_price_state_by_ticker(
        engine=resolved_engine,
        provider=provider,
        tickers=resolved_tickers,
    )
    quality_by_ticker = load_daily_price_quality_state_by_ticker(
        engine=resolved_engine,
        provider=provider,
        tickers=resolved_tickers,
        turnover_lookback_rows=QUALITY_TURNOVER_LOOKBACK_ROWS,
    )
    return TrainingEligibilityResult(
        provider=provider,
        tickers=resolved_tickers,
        states=tuple(
            _build_training_eligibility_state(
                ticker=ticker,
                stored_state=state_by_ticker.get(ticker),
                quality_state=quality_by_ticker.get(ticker),
            )
            for ticker in resolved_tickers
        ),
    )


def get_inference_ready_tickers(
    *,
    provider: str = yfinance_client.PROVIDER,
    reference_date: date | None = None,
    tickers: Sequence[str] | None = None,
    limit: int | None = None,
    config_dir: ConfigDir | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
) -> tuple[str, ...]:
    """Return only tickers currently ready for production inference.

    This is a convenience wrapper around ``check_inference_readiness`` that preserves the same
    ticker resolution and filtering semantics.
    """
    return check_inference_readiness(
        provider=provider,
        reference_date=reference_date,
        tickers=tickers,
        limit=limit,
        config_dir=config_dir,
        database_url=database_url,
        engine=engine,
    ).ready_tickers


def get_training_eligible_tickers(
    *,
    provider: str = yfinance_client.PROVIDER,
    tickers: Sequence[str] | None = None,
    limit: int | None = None,
    config_dir: ConfigDir | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
) -> tuple[str, ...]:
    """Return only tickers currently eligible for model training.

    This is a convenience wrapper around ``check_training_eligibility`` that preserves the
    same ticker resolution and filtering semantics.
    """
    return check_training_eligibility(
        provider=provider,
        tickers=tickers,
        limit=limit,
        config_dir=config_dir,
        database_url=database_url,
        engine=engine,
    ).eligible_tickers


def _build_inference_readiness_state(
    *,
    ticker: str,
    stored_state: BronzeDailyPriceState | None,
    quality_state: BronzeDailyPriceQualityState | None,
    reference_date: date,
) -> TickerInferenceReadinessState:
    if stored_state is None or quality_state is None:
        return TickerInferenceReadinessState(
            ticker=ticker,
            status=InferenceReadinessStatus.NOT_READY,
            failure_reasons=(EligibilityFailureReason.MISSING_BRONZE_DAILY_PRICES,),
            row_count=0,
            first_trading_date=None,
            last_trading_date=None,
            days_since_latest_row=None,
            missing_adjusted_close_count=0,
            null_or_zero_volume_count=0,
            null_or_zero_volume_ratio=None,
            latest_turnover_row_count=0,
            latest_median_turnover=None,
        )

    days_since_latest_row = (reference_date - stored_state.last_trading_date).days
    failure_reasons = _quality_failure_reasons(quality_state)
    if stored_state.row_count < INFERENCE_READINESS_MIN_DAILY_PRICE_ROWS:
        failure_reasons.append(EligibilityFailureReason.INSUFFICIENT_HISTORY)
    if days_since_latest_row > INFERENCE_READINESS_MAX_DAYS_SINCE_LATEST:
        failure_reasons.append(EligibilityFailureReason.STALE_DAILY_PRICES)

    return TickerInferenceReadinessState(
        ticker=ticker,
        status=_inference_status(failure_reasons),
        failure_reasons=tuple(failure_reasons),
        row_count=stored_state.row_count,
        first_trading_date=stored_state.first_trading_date,
        last_trading_date=stored_state.last_trading_date,
        days_since_latest_row=days_since_latest_row,
        missing_adjusted_close_count=quality_state.missing_adjusted_close_count,
        null_or_zero_volume_count=quality_state.null_or_zero_volume_count,
        null_or_zero_volume_ratio=_null_or_zero_volume_ratio(quality_state),
        latest_turnover_row_count=quality_state.latest_turnover_row_count,
        latest_median_turnover=quality_state.latest_median_turnover,
    )


def _build_training_eligibility_state(
    *,
    ticker: str,
    stored_state: BronzeDailyPriceState | None,
    quality_state: BronzeDailyPriceQualityState | None,
) -> TickerTrainingEligibilityState:
    if stored_state is None or quality_state is None:
        return TickerTrainingEligibilityState(
            ticker=ticker,
            status=TrainingEligibilityStatus.NOT_ELIGIBLE,
            failure_reasons=(EligibilityFailureReason.MISSING_BRONZE_DAILY_PRICES,),
            row_count=0,
            first_trading_date=None,
            last_trading_date=None,
            missing_adjusted_close_count=0,
            null_or_zero_volume_count=0,
            null_or_zero_volume_ratio=None,
            latest_turnover_row_count=0,
            latest_median_turnover=None,
        )

    failure_reasons = _quality_failure_reasons(quality_state)
    if stored_state.row_count < TRAINING_ELIGIBILITY_MIN_DAILY_PRICE_ROWS:
        failure_reasons.append(EligibilityFailureReason.INSUFFICIENT_HISTORY)

    return TickerTrainingEligibilityState(
        ticker=ticker,
        status=_training_status(failure_reasons),
        failure_reasons=tuple(failure_reasons),
        row_count=stored_state.row_count,
        first_trading_date=stored_state.first_trading_date,
        last_trading_date=stored_state.last_trading_date,
        missing_adjusted_close_count=quality_state.missing_adjusted_close_count,
        null_or_zero_volume_count=quality_state.null_or_zero_volume_count,
        null_or_zero_volume_ratio=_null_or_zero_volume_ratio(quality_state),
        latest_turnover_row_count=quality_state.latest_turnover_row_count,
        latest_median_turnover=quality_state.latest_median_turnover,
    )


def _quality_failure_reasons(
    quality_state: BronzeDailyPriceQualityState,
) -> list[EligibilityFailureReason]:
    failure_reasons: list[EligibilityFailureReason] = []
    if quality_state.missing_adjusted_close_count > 0:
        failure_reasons.append(EligibilityFailureReason.MISSING_ADJUSTED_CLOSE)
    if _null_or_zero_volume_ratio(quality_state) > QUALITY_MAX_NULL_OR_ZERO_VOLUME_RATIO:
        failure_reasons.append(EligibilityFailureReason.SPARSE_VOLUME)
    if quality_state.latest_turnover_row_count < QUALITY_MIN_TURNOVER_OBSERVATION_ROWS:
        failure_reasons.append(EligibilityFailureReason.INSUFFICIENT_TURNOVER_OBSERVATIONS)
    elif (
        quality_state.latest_median_turnover is None
        or quality_state.latest_median_turnover < QUALITY_MIN_MEDIAN_TURNOVER
    ):
        failure_reasons.append(EligibilityFailureReason.LOW_LIQUIDITY)
    return failure_reasons


def _null_or_zero_volume_ratio(quality_state: BronzeDailyPriceQualityState) -> Decimal:
    return Decimal(quality_state.null_or_zero_volume_count) / Decimal(quality_state.row_count)


def _inference_status(
    failure_reasons: list[EligibilityFailureReason],
) -> InferenceReadinessStatus:
    if failure_reasons:
        return InferenceReadinessStatus.NOT_READY
    return InferenceReadinessStatus.READY


def _training_status(
    failure_reasons: list[EligibilityFailureReason],
) -> TrainingEligibilityStatus:
    if failure_reasons:
        return TrainingEligibilityStatus.NOT_ELIGIBLE
    return TrainingEligibilityStatus.ELIGIBLE
