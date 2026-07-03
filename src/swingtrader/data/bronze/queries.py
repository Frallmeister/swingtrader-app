"""Read helpers for source-oriented bronze market data tables.

These helpers expose lightweight coverage summaries used by ingestion and operational jobs.
They do not transform bronze records into model-ready features.
"""

from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.engine import Engine

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
        Earliest stored trading date, or ``None`` when no rows are represented.
    last_trading_date
        Latest stored trading date, or ``None`` when no rows are represented.
    """

    ticker: str
    row_count: int
    first_trading_date: date | None
    last_trading_date: date | None


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
