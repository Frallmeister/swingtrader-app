"""Compare active tickers with bronze market data presence.

This module owns the bronze-data part of ticker onboarding. It treats the active
ticker configuration as desired production/trading state and the bronze daily price
table as actual stored-data state. Inference readiness and training eligibility are
separate checks in ``swingtrader.data.eligibility``.
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from sqlalchemy.engine import Engine

from swingtrader.core.db import resolve_database_engine
from swingtrader.data.bronze.queries import BronzeDailyPriceState, load_daily_price_state_by_ticker
from swingtrader.data.clients import yfinance as yfinance_client
from swingtrader.data.ingestion.market_data import IngestionResult, ingest_historical_daily_prices
from swingtrader.data.ingestion.universe_selection import ConfigDir, resolve_requested_tickers
from swingtrader.data.ingestion.validation import validate_date_window, validate_limit


class BronzeOnboardingStatus(StrEnum):
    """Bronze daily price onboarding status for one active ticker."""

    MISSING = "missing"
    ONBOARDED = "onboarded"


@dataclass(frozen=True)
class TickerBronzeOnboardingState:
    """Bronze daily price presence summary for one ticker."""

    ticker: str
    status: BronzeOnboardingStatus
    row_count: int
    first_trading_date: date | None
    last_trading_date: date | None


@dataclass(frozen=True)
class BronzeOnboardingResult:
    """Summary of active ticker presence in bronze daily prices."""

    provider: str
    tickers: tuple[str, ...]
    states: tuple[TickerBronzeOnboardingState, ...]

    @property
    def missing_count(self) -> int:
        return self._count_status(BronzeOnboardingStatus.MISSING)

    @property
    def onboarded_count(self) -> int:
        return self._count_status(BronzeOnboardingStatus.ONBOARDED)

    @property
    def missing_tickers(self) -> tuple[str, ...]:
        return tuple(
            item.ticker for item in self.states if item.status == BronzeOnboardingStatus.MISSING
        )

    def _count_status(self, status: BronzeOnboardingStatus) -> int:
        return sum(1 for item in self.states if item.status == status)


@dataclass(frozen=True)
class OnboardingSyncResult:
    """Summary of an active ticker bronze onboarding sync run."""

    start_date: date
    end_date: date
    onboarding_before: BronzeOnboardingResult
    backfill_tickers: tuple[str, ...]
    ingestion_result: IngestionResult | None
    onboarding_after: BronzeOnboardingResult | None

    @property
    def provider(self) -> str:
        return self.onboarding_before.provider

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.onboarding_before.tickers


def check_active_ticker_bronze_onboarding(
    *,
    provider: str = yfinance_client.PROVIDER,
    config_dir: ConfigDir | None = None,
    limit: int | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
) -> BronzeOnboardingResult:
    """Compare active tickers with bronze daily price presence.

    This function only decides whether an active ticker has any bronze daily price rows for
    the selected provider. It intentionally does not decide whether the ticker has enough
    history for inference or training.
    """
    validate_limit(limit)

    active_tickers = resolve_requested_tickers(config_dir=config_dir, limit=limit)
    resolved_engine = resolve_database_engine(database_url=database_url, engine=engine)
    presence_by_ticker = load_daily_price_state_by_ticker(
        engine=resolved_engine,
        provider=provider,
        tickers=active_tickers,
    )
    states = tuple(
        _build_ticker_onboarding_state(
            ticker=ticker,
            stored_presence=presence_by_ticker.get(ticker),
        )
        for ticker in active_tickers
    )
    return BronzeOnboardingResult(
        provider=provider,
        tickers=active_tickers,
        states=states,
    )


def sync_active_ticker_bronze_onboarding(
    *,
    start_date: date,
    end_date: date,
    provider: str = yfinance_client.PROVIDER,
    config_dir: ConfigDir | None = None,
    limit: int | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
    backfill: bool = False,
    raise_on_failure: bool = False,
) -> OnboardingSyncResult:
    """Report active ticker bronze presence and optionally backfill missing tickers."""
    validate_date_window(start_date=start_date, end_date=end_date)
    if backfill and provider != yfinance_client.PROVIDER:
        msg = f"Backfill is only supported for provider={yfinance_client.PROVIDER!r}."
        raise ValueError(msg)

    resolved_engine = resolve_database_engine(database_url=database_url, engine=engine)
    onboarding_before = check_active_ticker_bronze_onboarding(
        provider=provider,
        config_dir=config_dir,
        limit=limit,
        engine=resolved_engine,
    )
    backfill_tickers = onboarding_before.missing_tickers if backfill else ()
    ingestion_result = None
    onboarding_after = None
    if backfill_tickers:
        ingestion_result = ingest_historical_daily_prices(
            start_date=start_date,
            end_date=end_date,
            tickers=backfill_tickers,
            engine=resolved_engine,
            raise_on_failure=raise_on_failure,
        )
        onboarding_after = check_active_ticker_bronze_onboarding(
            provider=provider,
            config_dir=config_dir,
            limit=limit,
            engine=resolved_engine,
        )
    return OnboardingSyncResult(
        start_date=start_date,
        end_date=end_date,
        onboarding_before=onboarding_before,
        backfill_tickers=backfill_tickers,
        ingestion_result=ingestion_result,
        onboarding_after=onboarding_after,
    )


def _build_ticker_onboarding_state(
    *,
    ticker: str,
    stored_presence: BronzeDailyPriceState | None,
) -> TickerBronzeOnboardingState:
    if stored_presence is None:
        return TickerBronzeOnboardingState(
            ticker=ticker,
            status=BronzeOnboardingStatus.MISSING,
            row_count=0,
            first_trading_date=None,
            last_trading_date=None,
        )

    return TickerBronzeOnboardingState(
        ticker=ticker,
        status=BronzeOnboardingStatus.ONBOARDED,
        row_count=stored_presence.row_count,
        first_trading_date=stored_presence.first_trading_date,
        last_trading_date=stored_presence.last_trading_date,
    )
