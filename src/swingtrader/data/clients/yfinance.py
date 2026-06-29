"""Client helpers for downloading daily market data from yfinance."""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]
import yfinance as yf  # type: ignore[import-untyped]

PROVIDER = "yfinance"

DAILY_PRICE_COLUMNS = [
    "provider",
    "ticker",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "dividends",
    "stock_splits",
    "fetched_at",
    "request_id",
]

YFINANCE_FIELD_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adjusted_close": "Adj Close",
    "volume": "Volume",
    "dividends": "Dividends",
    "stock_splits": "Stock Splits",
}

logger = logging.getLogger(__name__)


def download_daily_prices(
    tickers: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    fetched_at: datetime | None = None,
    request_id: str | None = None,
) -> pd.DataFrame:
    """Download daily prices from yfinance and return bronze-shaped rows."""
    normalized_tickers = _normalize_tickers(tickers)
    resolved_fetched_at = _normalize_fetched_at(fetched_at or datetime.now(UTC))
    resolved_request_id = request_id or str(uuid4())

    logger.info(
        "Downloading daily prices provider=%s tickers=%s start_date=%s end_date=%s request_id=%s",
        PROVIDER,
        len(normalized_tickers),
        start_date,
        end_date,
        resolved_request_id,
    )
    raw_prices = yf.download(
        tickers=list(normalized_tickers),
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        group_by="ticker",
        actions=True,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    prices = normalize_daily_prices(
        raw_prices,
        tickers=normalized_tickers,
        fetched_at=resolved_fetched_at,
        request_id=resolved_request_id,
    )
    logger.info(
        "Downloaded daily prices provider=%s tickers=%s rows=%s request_id=%s",
        PROVIDER,
        len(normalized_tickers),
        len(prices),
        resolved_request_id,
    )
    return prices


def normalize_daily_prices(
    raw_prices: pd.DataFrame,
    *,
    tickers: Sequence[str],
    fetched_at: datetime,
    request_id: str,
) -> pd.DataFrame:
    """Normalize a yfinance daily price DataFrame into bronze-shaped rows."""
    normalized_tickers = _normalize_tickers(tickers)
    normalized_fetched_at = _normalize_fetched_at(fetched_at)
    if raw_prices.empty:
        return _empty_daily_prices()

    rows = []
    for ticker in normalized_tickers:
        ticker_prices = _select_ticker_prices(raw_prices=raw_prices, ticker=ticker)
        if ticker_prices is None or ticker_prices.empty:
            continue
        rows.append(
            _normalize_ticker_prices(
                ticker_prices=ticker_prices,
                ticker=ticker,
                fetched_at=normalized_fetched_at,
                request_id=request_id,
            )
        )

    if not rows:
        return _empty_daily_prices()

    prices = pd.concat(rows, ignore_index=True)
    prices = prices.dropna(
        how="all",
        subset=[
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "dividends",
            "stock_splits",
        ],
    )
    return (
        prices[DAILY_PRICE_COLUMNS].sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    )


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    # dict.fromkeys preserves first-seen order while removing duplicates.
    normalized_tickers = tuple(
        dict.fromkeys(ticker.strip() for ticker in tickers if ticker.strip())
    )
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")
    return normalized_tickers


def _normalize_fetched_at(fetched_at: datetime) -> datetime:
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        raise ValueError("fetched_at must be timezone-aware.")
    return fetched_at.astimezone(UTC)


def _select_ticker_prices(raw_prices: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if not isinstance(raw_prices.columns, pd.MultiIndex):
        return raw_prices.copy()

    if ticker in raw_prices.columns.get_level_values(0):
        return raw_prices[ticker].copy()
    if ticker in raw_prices.columns.get_level_values(1):
        return raw_prices.xs(ticker, axis=1, level=1).copy()
    return None


def _normalize_ticker_prices(
    ticker_prices: pd.DataFrame,
    *,
    ticker: str,
    fetched_at: datetime,
    request_id: str,
) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "provider": PROVIDER,
            "ticker": ticker,
            "trading_date": pd.to_datetime(ticker_prices.index).date,
            "fetched_at": fetched_at,
            "request_id": request_id,
        }
    )
    for output_column, source_column in YFINANCE_FIELD_MAP.items():
        normalized[output_column] = _get_optional_column(ticker_prices, source_column)
    return normalized


def _get_optional_column(ticker_prices: pd.DataFrame, source_column: str) -> pd.Series:
    if source_column in ticker_prices:
        return ticker_prices[source_column].reset_index(drop=True)
    return pd.Series([pd.NA] * len(ticker_prices))


def _empty_daily_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_PRICE_COLUMNS)
