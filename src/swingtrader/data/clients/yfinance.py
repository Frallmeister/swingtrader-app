"""Client helpers for downloading daily market data from yfinance."""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime
from uuid import uuid4

import pandas as pd
import yfinance as yf

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
YFINANCE_TO_BRONZE_COLUMNS = {value: key for key, value in YFINANCE_FIELD_MAP.items()}

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

    ticker_first_prices = _to_ticker_first_prices(
        raw_prices=raw_prices,
        tickers=normalized_tickers,
    )
    if ticker_first_prices.empty:
        return _empty_daily_prices()

    prices = (
        ticker_first_prices.stack(level=0)
        .rename_axis(index=["trading_date", "ticker"], columns=None)
        .reset_index()
        .rename(columns=YFINANCE_TO_BRONZE_COLUMNS)
    )
    prices["trading_date"] = pd.to_datetime(prices["trading_date"]).dt.date
    prices["provider"] = PROVIDER
    prices["fetched_at"] = normalized_fetched_at
    prices["request_id"] = request_id
    for output_column in YFINANCE_FIELD_MAP:
        if output_column not in prices:
            prices[output_column] = pd.NA

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
    normalized_prices = (
        prices[DAILY_PRICE_COLUMNS].sort_values(["ticker", "trading_date"]).reset_index(drop=True)
    )
    return normalized_prices


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


def _to_ticker_first_prices(raw_prices: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Return prices with MultiIndex columns ordered as ticker, then yfinance field."""
    if not isinstance(raw_prices.columns, pd.MultiIndex):
        if len(tickers) > 1:
            msg = "Non-hierarchical yfinance data can only be normalized for one ticker."
            raise ValueError(msg)
        return pd.concat({tickers[0]: raw_prices}, axis=1)

    # yfinance may return columns as either (ticker, field) or (field, ticker).
    # Find the level containing requested tickers, then swap only field-first data.
    for ticker_level in (0, 1):
        ticker_mask = raw_prices.columns.get_level_values(ticker_level).isin(tickers)
        if ticker_mask.any():
            ticker_prices = raw_prices.loc[:, ticker_mask].copy()
            if ticker_level == 0:
                return ticker_prices
            return ticker_prices.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

    return raw_prices.iloc[:, 0:0].copy()


def _empty_daily_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=DAILY_PRICE_COLUMNS)
