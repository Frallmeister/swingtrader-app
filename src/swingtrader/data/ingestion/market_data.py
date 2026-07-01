"""Orchestrate historical market data ingestion into bronze storage.

This module resolves requested tickers, delegates provider downloads to the yfinance client,
and delegates idempotent persistence to the bronze writer.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from uuid import uuid4

from sqlalchemy.engine import Engine

from swingtrader.core.db import create_database_engine, initialize_database
from swingtrader.data.bronze.writer import upsert_daily_prices
from swingtrader.data.clients import yfinance as yfinance_client
from swingtrader.data.ingestion.universe_selection import resolve_active_tickers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TickerIngestionFailure:
    """Failure details for one ticker in an ingestion run."""

    ticker: str
    error_type: str
    message: str


@dataclass(frozen=True)
class IngestionResult:
    """Summary of a historical market data ingestion run."""

    provider: str
    request_id: str
    start_date: date
    end_date: date
    tickers: tuple[str, ...]
    downloaded_rows: int
    upserted_rows: int
    failures: tuple[TickerIngestionFailure, ...]


class MarketDataIngestionError(RuntimeError):
    """Raised when an ingestion run has failures and strict failure handling is enabled."""

    def __init__(self, result: IngestionResult):
        self.result = result
        super().__init__(
            f"Historical market data ingestion failed for {len(result.failures)} ticker(s)."
        )


def ingest_historical_daily_prices(
    *,
    start_date: date,
    end_date: date,
    tickers: Sequence[str] | None = None,
    limit: int | None = None,
    database_url: str | None = None,
    engine: Engine | None = None,
    fetched_at: datetime | None = None,
    request_id: str | None = None,
    raise_on_failure: bool = False,
) -> IngestionResult:
    """Download historical daily prices and upsert them into bronze storage.

    Parameters
    ----------
    start_date
        Inclusive start date passed to the market data provider.
    end_date
        Exclusive end date passed to the market data provider. Must be later than
        ``start_date``.
    tickers
        Optional ticker symbols to ingest. When omitted, active tickers are resolved from the
        packaged universe configuration.
    limit
        Optional maximum number of resolved tickers to ingest. This is intended for small
        smoke runs and is applied after explicit or active tickers have been normalized.
    database_url
        Optional SQLAlchemy database URL. Mutually exclusive with ``engine``.
    engine
        Optional SQLAlchemy engine. Passing an engine is useful for tests and callers that
        already manage database connections. Mutually exclusive with ``database_url``.
    fetched_at
        Timestamp recorded on all downloaded rows. When omitted, the current UTC time is used.
    request_id
        Provenance identifier recorded on all downloaded rows and returned in the result.
        When omitted, a UUID is generated.
    raise_on_failure
        When ``False``, per-ticker failures are recorded in the result and ingestion
        continues. When ``True``, ``MarketDataIngestionError`` is raised after all tickers
        have been attempted if any ticker failed.

    Returns
    -------
    IngestionResult
        Summary of the ingestion run, including provider, request id, requested date range,
        attempted tickers, row counts, and per-ticker failures.

    Raises
    ------
    ValueError
        Raised when the date range is invalid, both ``database_url`` and ``engine`` are
        provided, ``limit`` is less than one, or no tickers resolve for ingestion.
    MarketDataIngestionError
        Raised when at least one ticker failed and ``raise_on_failure`` is ``True``. The
        exception carries the partial ``IngestionResult`` on ``.result``.

    Notes
    -----
    Tickers are ingested one at a time so a provider or write failure for one ticker does not
    prevent successful tickers from being written. The database schema is initialized before
    ingestion begins.
    """
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date.")
    if engine is not None and database_url is not None:
        raise ValueError("Pass either engine or database_url, not both.")

    resolved_tickers = _resolve_requested_tickers(tickers=tickers, limit=limit)
    resolved_fetched_at = fetched_at or datetime.now(UTC)
    resolved_request_id = request_id or str(uuid4())
    resolved_engine = engine or create_database_engine(database_url)
    initialize_database(resolved_engine)

    logger.info(
        "Starting historical daily price ingestion provider=%s tickers=%s start_date=%s "
        "end_date=%s request_id=%s",
        yfinance_client.PROVIDER,
        len(resolved_tickers),
        start_date,
        end_date,
        resolved_request_id,
    )

    downloaded_rows = 0
    upserted_rows = 0
    failures: list[TickerIngestionFailure] = []
    for ticker in resolved_tickers:
        try:
            prices = yfinance_client.download_daily_prices(
                tickers=[ticker],
                start_date=start_date,
                end_date=end_date,
                fetched_at=resolved_fetched_at,
                request_id=resolved_request_id,
            )
            downloaded_rows += len(prices)
            upserted_rows += upsert_daily_prices(prices=prices, engine=resolved_engine)
        except Exception as exc:
            failures.append(
                TickerIngestionFailure(
                    ticker=ticker,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            logger.exception(
                "Failed historical daily price ingestion ticker=%s provider=%s start_date=%s "
                "end_date=%s request_id=%s",
                ticker,
                yfinance_client.PROVIDER,
                start_date,
                end_date,
                resolved_request_id,
            )

    result = IngestionResult(
        provider=yfinance_client.PROVIDER,
        request_id=resolved_request_id,
        start_date=start_date,
        end_date=end_date,
        tickers=resolved_tickers,
        downloaded_rows=downloaded_rows,
        upserted_rows=upserted_rows,
        failures=tuple(failures),
    )
    logger.info(
        "Finished historical daily price ingestion provider=%s tickers=%s downloaded_rows=%s "
        "upserted_rows=%s failures=%s request_id=%s",
        result.provider,
        len(result.tickers),
        result.downloaded_rows,
        result.upserted_rows,
        len(result.failures),
        result.request_id,
    )
    if result.failures and raise_on_failure:
        raise MarketDataIngestionError(result)
    return result


def _resolve_requested_tickers(
    *,
    tickers: Sequence[str] | None,
    limit: int | None,
) -> tuple[str, ...]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero.")

    requested_tickers = tickers if tickers is not None else resolve_active_tickers()
    normalized_tickers = tuple(
        dict.fromkeys(ticker.strip() for ticker in requested_tickers if ticker.strip())
    )
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")
    if limit is None:
        return normalized_tickers
    return normalized_tickers[:limit]
