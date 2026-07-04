"""Read helpers for source-oriented bronze market data tables.

These helpers expose lightweight coverage summaries used by ingestion and operational jobs.
They do not transform bronze records into model-ready features.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from statistics import median

from sqlalchemy import case, func, or_, select
from sqlalchemy.engine import Engine, RowMapping

from swingtrader.data.bronze.schema import bronze_market_daily_prices


@dataclass(frozen=True)
class BronzeDailyPriceState:
    """Stored daily price coverage summary for one ticker and provider.

    Attributes
    ----------
    ticker
        Provider ticker symbol.
    row_count
        Number of bronze daily price rows stored for the provider and ticker.
    first_trading_date
        Earliest stored trading date.
    last_trading_date
        Latest stored trading date.
    """

    ticker: str
    row_count: int
    first_trading_date: date
    last_trading_date: date


@dataclass(frozen=True)
class BronzeDailyPriceQualityState:
    """Stored daily price quality summary for one ticker and provider.

    The summary remains bronze-oriented: it describes source row completeness and recent
    traded value, but it does not decide readiness or transform rows into model features.
    """

    ticker: str
    row_count: int
    missing_adjusted_close_count: int
    null_or_zero_volume_count: int
    latest_turnover_row_count: int
    latest_median_turnover: Decimal | None


def load_daily_price_state_by_ticker(
    *,
    engine: Engine,
    provider: str,
    tickers: tuple[str, ...],
) -> dict[str, BronzeDailyPriceState]:
    """Load stored bronze daily price coverage for requested tickers.

    Tickers without stored rows are omitted from the returned mapping.

    Parameters
    ----------
    engine
        SQLAlchemy engine for the target application database.
    provider
        Market data provider to filter by, such as ``"yfinance"``.
    tickers
        Requested ticker symbols.

    Returns
    -------
    dict[str, BronzeDailyPriceState]
        Coverage state keyed by ticker. Missing tickers are absent rather than represented by
        zero-row objects.
    """
    if not tickers:
        return {}

    statement = (
        select(
            bronze_market_daily_prices.c.ticker,
            func.count().label("row_count"),
            func.min(bronze_market_daily_prices.c.trading_date).label("first_trading_date"),
            func.max(bronze_market_daily_prices.c.trading_date).label("last_trading_date"),
        )
        .where(bronze_market_daily_prices.c.provider == provider)
        .where(bronze_market_daily_prices.c.ticker.in_(tickers))
        .group_by(bronze_market_daily_prices.c.ticker)
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).mappings().all()
    return {
        str(row["ticker"]): BronzeDailyPriceState(
            ticker=str(row["ticker"]),
            row_count=int(row["row_count"]),
            first_trading_date=row["first_trading_date"],
            last_trading_date=row["last_trading_date"],
        )
        for row in rows
    }


def load_daily_price_quality_state_by_ticker(
    *,
    engine: Engine,
    provider: str,
    tickers: tuple[str, ...],
    turnover_lookback_rows: int = 60,
) -> dict[str, BronzeDailyPriceQualityState]:
    """Load bronze daily price quality summaries for requested tickers.

    Tickers without stored rows are omitted from the returned mapping. Median turnover is
    computed from the latest ``turnover_lookback_rows`` rows in Python for database
    portability across SQLite and PostgreSQL.
    """
    if not tickers:
        return {}
    if turnover_lookback_rows < 1:
        raise ValueError("turnover_lookback_rows must be greater than zero.")

    aggregate_statement = (
        select(
            bronze_market_daily_prices.c.ticker,
            func.count().label("row_count"),
            func.sum(
                case((bronze_market_daily_prices.c.adjusted_close.is_(None), 1), else_=0)
            ).label("missing_adjusted_close_count"),
            func.sum(
                case(
                    (
                        or_(
                            bronze_market_daily_prices.c.volume.is_(None),
                            bronze_market_daily_prices.c.volume <= 0,
                        ),
                        1,
                    ),
                    else_=0,
                )
            ).label("null_or_zero_volume_count"),
        )
        .where(bronze_market_daily_prices.c.provider == provider)
        .where(bronze_market_daily_prices.c.ticker.in_(tickers))
        .group_by(bronze_market_daily_prices.c.ticker)
    )
    turnover_statement = (
        select(
            bronze_market_daily_prices.c.ticker,
            bronze_market_daily_prices.c.close,
            bronze_market_daily_prices.c.volume,
        )
        .where(bronze_market_daily_prices.c.provider == provider)
        .where(bronze_market_daily_prices.c.ticker.in_(tickers))
        .order_by(
            bronze_market_daily_prices.c.ticker,
            bronze_market_daily_prices.c.trading_date.desc(),
        )
    )
    with engine.connect() as connection:
        aggregate_rows = connection.execute(aggregate_statement).mappings().all()
        turnover_rows = connection.execute(turnover_statement).mappings().all()

    turnover_values_by_ticker = _latest_turnover_values_by_ticker(
        rows=turnover_rows,
        turnover_lookback_rows=turnover_lookback_rows,
    )
    return {
        str(row["ticker"]): BronzeDailyPriceQualityState(
            ticker=str(row["ticker"]),
            row_count=int(row["row_count"]),
            missing_adjusted_close_count=int(row["missing_adjusted_close_count"] or 0),
            null_or_zero_volume_count=int(row["null_or_zero_volume_count"] or 0),
            latest_turnover_row_count=len(turnover_values_by_ticker[str(row["ticker"])]),
            latest_median_turnover=_median_or_none(turnover_values_by_ticker[str(row["ticker"])]),
        )
        for row in aggregate_rows
    }


def _latest_turnover_values_by_ticker(
    *,
    rows: Sequence[RowMapping],
    turnover_lookback_rows: int,
) -> dict[str, list[Decimal]]:
    values_by_ticker: dict[str, list[Decimal]] = defaultdict(list)
    seen_by_ticker: dict[str, int] = defaultdict(int)
    for row in rows:
        ticker = str(row["ticker"])
        if seen_by_ticker[ticker] >= turnover_lookback_rows:
            continue
        seen_by_ticker[ticker] += 1

        close = _decimal_or_none(row["close"])
        volume = _positive_int_or_none(row["volume"])
        if close is None or volume is None:
            continue
        values_by_ticker[ticker].append(close * Decimal(volume))
    return values_by_ticker


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _positive_int_or_none(value: object) -> int | None:
    if value is None:
        return None
    parsed_value = int(Decimal(str(value)))
    if parsed_value <= 0:
        return None
    return parsed_value


def _median_or_none(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return median(values)
