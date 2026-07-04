"""Run initial bronze market data onboarding for missing active tickers.

This job creates the first bronze daily price rows for active tickers that do not yet
have any stored market data. The daily update job remains responsible for keeping
already-onboarded active tickers current.
"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.engine import Engine

from swingtrader.core.db import resolve_database_engine
from swingtrader.core.logging_config import configure_logging
from swingtrader.data.ingestion.market_data import IngestionResult, TickerIngestionFailure
from swingtrader.data.ingestion.market_data_settings import ConfigFile, load_market_data_settings
from swingtrader.data.ingestion.onboarding import (
    BronzeOnboardingResult,
    BronzeOnboardingStatus,
    OnboardingSyncResult,
    sync_active_ticker_bronze_onboarding,
)
from swingtrader.data.ingestion.universe_selection import ConfigDir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnboardMarketDataResult:
    """Summary of an initial market data onboarding job run."""

    provider: str
    start_date: date
    end_date: date
    active_tickers: tuple[str, ...]
    already_onboarded_tickers: tuple[str, ...]
    missing_tickers: tuple[str, ...]
    attempted_tickers: tuple[str, ...]
    ingestion_result: IngestionResult | None
    onboarding_after: BronzeOnboardingResult | None

    @property
    def successful_tickers(self) -> tuple[str, ...]:
        failed_tickers = set(self.failed_tickers)
        return tuple(ticker for ticker in self.attempted_tickers if ticker not in failed_tickers)

    @property
    def failed_tickers(self) -> tuple[str, ...]:
        return tuple(failure.ticker for failure in self.failures)

    @property
    def downloaded_rows(self) -> int:
        if self.ingestion_result is None:
            return 0
        return self.ingestion_result.downloaded_rows

    @property
    def upserted_rows(self) -> int:
        if self.ingestion_result is None:
            return 0
        return self.ingestion_result.upserted_rows

    @property
    def failures(self) -> tuple[TickerIngestionFailure, ...]:
        if self.ingestion_result is None:
            return ()
        return self.ingestion_result.failures


def run_onboard_market_data(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
    config_dir: ConfigDir | None = None,
    settings_path: ConfigFile | None = None,
) -> OnboardMarketDataResult:
    """Run initial market data onboarding for missing active tickers.

    Tickers with existing bronze daily price rows are skipped. Missing active tickers are
    backfilled from ``start_date`` to the exclusive ``end_date`` through the existing
    historical ingestion path.
    """
    settings = load_market_data_settings(settings_path)
    resolved_start_date = start_date or settings.initial_start_date
    resolved_end_date = end_date or _default_end_date()
    resolved_engine = resolve_database_engine(database_url=database_url, engine=engine)

    logger.info(
        "Starting market data onboarding provider=%s start_date=%s end_date=%s limit=%s",
        settings.provider,
        resolved_start_date,
        resolved_end_date,
        limit,
    )
    sync_result = sync_active_ticker_bronze_onboarding(
        start_date=resolved_start_date,
        end_date=resolved_end_date,
        provider=settings.provider,
        config_dir=config_dir,
        limit=limit,
        engine=resolved_engine,
        backfill=True,
        raise_on_failure=False,
    )
    result = _build_onboard_market_data_result(sync_result)
    if result.active_tickers and not result.missing_tickers:
        logger.info(
            "Market data onboarding is a no-op because all active tickers are already "
            "onboarded provider=%s active_tickers=%s",
            result.provider,
            len(result.active_tickers),
        )

    logger.info(
        "Finished market data onboarding provider=%s active_tickers=%s "
        "already_onboarded_tickers=%s missing_tickers=%s attempted_tickers=%s "
        "successful_tickers=%s failed_tickers=%s downloaded_rows=%s upserted_rows=%s",
        result.provider,
        len(result.active_tickers),
        len(result.already_onboarded_tickers),
        len(result.missing_tickers),
        len(result.attempted_tickers),
        len(result.successful_tickers),
        len(result.failed_tickers),
        result.downloaded_rows,
        result.upserted_rows,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the market data onboarding command-line interface."""
    args = _parse_args(argv)
    configure_logging()
    result = run_onboard_market_data(
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        database_url=args.database_url,
    )
    if args.fail_on_ticker_failure and result.failures:
        return 1
    return 0


def _build_onboard_market_data_result(sync_result: OnboardingSyncResult) -> OnboardMarketDataResult:
    onboarding_before = sync_result.onboarding_before
    return OnboardMarketDataResult(
        provider=sync_result.provider,
        start_date=sync_result.start_date,
        end_date=sync_result.end_date,
        active_tickers=sync_result.tickers,
        already_onboarded_tickers=_tickers_with_status(
            onboarding_before,
            BronzeOnboardingStatus.ONBOARDED,
        ),
        missing_tickers=_tickers_with_status(
            onboarding_before,
            BronzeOnboardingStatus.MISSING,
        ),
        attempted_tickers=sync_result.backfill_tickers,
        ingestion_result=sync_result.ingestion_result,
        onboarding_after=sync_result.onboarding_after,
    )


def _tickers_with_status(
    onboarding: BronzeOnboardingResult,
    status: BronzeOnboardingStatus,
) -> tuple[str, ...]:
    return tuple(item.ticker for item in onboarding.states if item.status == status)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Onboard initial bronze daily market data.")
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help=(
            "Inclusive start date for provider requests, in YYYY-MM-DD format. "
            "Defaults to market_data.yml initial_start_date."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help=(
            "Exclusive end date for provider requests, in YYYY-MM-DD format. "
            "Defaults to tomorrow UTC."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of active tickers for smoke runs.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "Explicit SQLAlchemy database URL. Defaults to SWINGTRADER_DATABASE_URL "
            "or local SQLite."
        ),
    )
    parser.add_argument(
        "--fail-on-ticker-failure",
        action="store_true",
        help="Exit with status 1 if any ticker fails while still writing successful tickers.",
    )
    return parser.parse_args(argv)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"Invalid date {value!r}; expected YYYY-MM-DD."
        raise argparse.ArgumentTypeError(msg) from exc


def _default_end_date() -> date:
    return datetime.now(UTC).date() + timedelta(days=1)


if __name__ == "__main__":
    raise SystemExit(main())
