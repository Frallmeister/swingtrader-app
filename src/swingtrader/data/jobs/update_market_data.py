"""Run daily bronze market data updates for the active trading universe.

This module is the first runnable data job for the project. It resolves the active ticker
configuration, reads existing bronze daily price coverage, derives per-ticker update
plans, and delegates provider download and idempotent write behavior to the existing
historical market data ingestion path.
"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from sqlalchemy.engine import Engine

from swingtrader.core.logging_config import configure_logging
from swingtrader.data.bronze.queries import BronzeDailyPriceState, load_daily_price_state_by_ticker
from swingtrader.data.db import resolve_database_engine
from swingtrader.data.ingestion.market_data import (
    IngestionResult,
    TickerIngestionFailure,
    ingest_historical_daily_prices,
)
from swingtrader.data.ingestion.market_data_settings import (
    ConfigFile,
    load_market_data_settings,
)
from swingtrader.data.ingestion.universe_selection import ConfigDir, resolve_requested_tickers
from swingtrader.data.ingestion.validation import validate_limit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyMarketDataPlannedUpdate:
    """Planned provider request for one active ticker.

    The job plans one update per ticker because the current historical ingestion path calls
    the market data provider one ticker at a time.
    """

    start_date: date
    end_date: date
    ticker: str


@dataclass(frozen=True)
class DailyMarketDataUpdateResult:
    """Summary of a daily market data update job run.

    The result aggregates all per-ticker ingestion calls made by the job and exposes combined
    row counts and ticker-level failures for logging, tests, and CLI exit-code decisions.
    """

    provider: str
    end_date: date
    active_tickers: tuple[str, ...]
    not_onboarded_tickers: tuple[str, ...]
    skipped_tickers: tuple[str, ...]
    planned_updates: tuple[DailyMarketDataPlannedUpdate, ...]
    ingestion_results: tuple[IngestionResult, ...]

    @property
    def update_tickers(self) -> tuple[str, ...]:
        """Tickers that were selected for daily refresh attempts."""
        return tuple(update.ticker for update in self.planned_updates)

    @property
    def downloaded_rows(self) -> int:
        """Total normalized daily price rows returned by daily update downloads."""
        return sum(result.downloaded_rows for result in self.ingestion_results)

    @property
    def upserted_rows(self) -> int:
        """Total bronze daily price rows submitted to idempotent upserts."""
        return sum(result.upserted_rows for result in self.ingestion_results)

    @property
    def failures(self) -> tuple[TickerIngestionFailure, ...]:
        """Per-ticker ingestion failures from attempted daily refreshes.

        Tickers reported in ``not_onboarded_tickers`` are not attempted by this job and are
        therefore not represented as failures here.
        """
        return tuple(failure for result in self.ingestion_results for failure in result.failures)


def run_daily_market_data_update(
    *,
    end_date: date | None = None,
    limit: int | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
    config_dir: ConfigDir | None = None,
    settings_path: ConfigFile | None = None,
) -> DailyMarketDataUpdateResult:
    """Run the daily market data update workflow.

    The job derives planned updates from stored bronze daily price state. Tickers without
    stored rows are reported as not onboarded and omitted from daily update planning. Tickers
    with existing rows overlap from their latest stored trading date and rely on idempotent
    bronze upserts.

    Parameters
    ----------
    end_date
        Exclusive end date for provider requests. When omitted, tomorrow's UTC date is used.
    limit
        Optional maximum number of resolved active tickers to update. Intended for smoke runs.
    database_url
        Optional SQLAlchemy database URL. Mutually exclusive with ``engine``.
    engine
        Optional SQLAlchemy engine. Passing an engine is useful for tests and callers that
        already manage database connections. Mutually exclusive with ``database_url``.
    config_dir
        Optional active ticker universe configuration directory. When omitted, packaged
        universe configuration is used.
    settings_path
        Optional market data settings file. When omitted, packaged project settings are used.

    Returns
    -------
    DailyMarketDataUpdateResult
        Aggregated job result including active tickers, skipped tickers, planned updates,
        not-onboarded tickers, and ingestion results.

    Notes
    -----
    This workflow updates only already-onboarded active tickers. It does not use
    ``MarketDataSettings.initial_start_date`` and does not create the first bronze rows for a
    ticker; use the onboarding job for initial fills. This function does not configure logging
    and does not call ``sys.exit``. Use ``main`` for the command-line entrypoint.
    """
    validate_limit(limit)

    settings = load_market_data_settings(settings_path)
    resolved_end_date = end_date or _default_end_date()
    resolved_engine = resolve_database_engine(database_url=database_url, engine=engine)
    active_tickers = resolve_requested_tickers(config_dir=config_dir, limit=limit)

    logger.info(
        "Starting daily market data update provider=%s active_tickers=%s end_date=%s",
        settings.provider,
        len(active_tickers),
        resolved_end_date,
    )

    state_by_ticker = load_daily_price_state_by_ticker(
        engine=resolved_engine,
        provider=settings.provider,
        tickers=active_tickers,
    )
    not_onboarded_tickers = tuple(
        ticker for ticker in active_tickers if ticker not in state_by_ticker
    )
    planned_updates = plan_daily_market_data_updates(
        active_tickers=active_tickers,
        state_by_ticker=state_by_ticker,
        end_date=resolved_end_date,
    )
    if active_tickers and not planned_updates and len(not_onboarded_tickers) == len(active_tickers):
        logger.warning(
            "Daily market data update has no planned updates because no active tickers are "
            "onboarded provider=%s active_tickers=%s end_date=%s",
            settings.provider,
            len(active_tickers),
            resolved_end_date,
        )
    ingestion_results = []
    for update in planned_updates:
        logger.info(
            "Updating daily market data ticker provider=%s ticker=%s start_date=%s end_date=%s",
            settings.provider,
            update.ticker,
            update.start_date,
            update.end_date,
        )
        ingestion_results.append(
            ingest_historical_daily_prices(
                start_date=update.start_date,
                end_date=update.end_date,
                tickers=(update.ticker,),
                engine=resolved_engine,
                raise_on_failure=False,
            )
        )
    update_tickers = {update.ticker for update in planned_updates}
    skipped_tickers = tuple(
        ticker
        for ticker in active_tickers
        if ticker in state_by_ticker and ticker not in update_tickers
    )
    result = DailyMarketDataUpdateResult(
        provider=settings.provider,
        end_date=resolved_end_date,
        active_tickers=active_tickers,
        not_onboarded_tickers=not_onboarded_tickers,
        skipped_tickers=skipped_tickers,
        planned_updates=planned_updates,
        ingestion_results=tuple(ingestion_results),
    )
    logger.info(
        "Finished daily market data update provider=%s active_tickers=%s update_tickers=%s "
        "not_onboarded_tickers=%s skipped_tickers=%s planned_updates=%s downloaded_rows=%s "
        "upserted_rows=%s failures=%s",
        result.provider,
        len(result.active_tickers),
        len(result.update_tickers),
        len(result.not_onboarded_tickers),
        len(result.skipped_tickers),
        len(result.planned_updates),
        result.downloaded_rows,
        result.upserted_rows,
        len(result.failures),
    )
    return result


def plan_daily_market_data_updates(
    *,
    active_tickers: tuple[str, ...],
    state_by_ticker: dict[str, BronzeDailyPriceState],
    end_date: date,
) -> tuple[DailyMarketDataPlannedUpdate, ...]:
    """Build per-ticker update plans from active tickers and bronze state.

    Parameters
    ----------
    active_tickers
        Tickers in the active production/trading universe.
    state_by_ticker
        Stored bronze daily price state keyed by ticker. Tickers missing from the mapping are
        treated as not onboarded and omitted from daily update planning.
    end_date
        Exclusive end date for provider requests.

    Returns
    -------
    tuple[DailyMarketDataPlannedUpdate, ...]
        Planned updates in active ticker order. Tickers whose computed start date is not before
        ``end_date`` are omitted.

    Notes
    -----
    Existing tickers start from their latest stored trading date rather than the following day.
    That one-row overlap is intentional: bronze writes are idempotent, and the overlap lets
    reruns pick up provider corrections for the latest stored observation.
    """
    planned_updates: list[DailyMarketDataPlannedUpdate] = []
    for ticker in active_tickers:
        state = state_by_ticker.get(ticker)
        if state is None:
            continue
        start_date = state.last_trading_date
        if start_date >= end_date:
            continue
        planned_updates.append(
            DailyMarketDataPlannedUpdate(
                start_date=start_date,
                end_date=end_date,
                ticker=ticker,
            )
        )

    return tuple(planned_updates)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the daily market data update command-line interface.

    Parameters
    ----------
    argv
        Optional argument sequence. When omitted, arguments are read from ``sys.argv`` by
        ``argparse``.

    Returns
    -------
    int
        Process-style exit code. Returns ``1`` when ``--fail-on-ticker-failure`` is passed and
        at least one ticker failed; otherwise returns ``0``.
    """
    args = _parse_args(argv)
    configure_logging()
    result = run_daily_market_data_update(
        end_date=args.end_date,
        limit=args.limit,
        database_url=args.database_url,
    )
    if args.fail_on_ticker_failure and result.failures:
        return 1
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update bronze daily market data.")
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


def _default_end_date() -> date:
    return datetime.now(UTC).date() + timedelta(days=1)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"Invalid date {value!r}. Expected YYYY-MM-DD."
        raise argparse.ArgumentTypeError(msg) from exc


if __name__ == "__main__":
    raise SystemExit(main())
