"""Volatility feature transformations for daily high/low/close price histories.

This module builds row-aligned, point-in-time volatility features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each smoothing window has
enough prior observations.

Numerical indicators accept either one ordered dataframe for a single ticker or
a multi-ticker dataframe carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels. In the latter case the calculation is applied
independently within each provider/ticker group and the input row order is
preserved. Because volatility indicators consume several price columns they take
a dataframe rather than a single series, and each returns one index-aligned
series. The family orchestrator returns a copy of the input dataframe with the
final model feature columns appended. The module currently implements True
Range and ATR-based features.
"""

import pandas as pd

from swingtrader.data.features._numerical import (
    safe_divide,
    wilder_moving_average,
)
from swingtrader.data.features._validation import (
    apply_by_ticker,
    validate_length,
    validate_market_price_index,
    validate_required_columns,
)

_REQUIRED_PRICE_COLUMNS = frozenset({"high", "low", "close"})


def add_volatility_features(
    data: pd.DataFrame,
    *,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Return a copy of data with the default volatility feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus
    ``high``, ``low``, and ``close`` columns. The index must be unique and
    sorted. The returned dataframe preserves the input rows and appends the final
    ATR-percent feature column.

    Raw True Range and ATR are expressed in the input price units and are not
    comparable across tickers, so the orchestrator only appends the
    scale-invariant ``atr_percent`` column. Use :func:`true_range` and
    :func:`atr` directly when absolute price-unit values are required.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns=_REQUIRED_PRICE_COLUMNS)
    validate_length(atr_length)

    data = data.copy()
    data["atr_percent"] = atr_percent(data, length=atr_length)
    return data


def true_range(data: pd.DataFrame) -> pd.Series:
    """Calculate the True Range for one or many tickers.

    ``data`` must contain ``high``, ``low``, and ``close`` columns in
    chronological order. True Range is the greatest of the current high-low
    range, the absolute gap between the current high and the previous close, and
    the absolute gap between the current low and the previous close. The first
    row of each ticker has no previous close, so its True Range falls back to the
    current high-low range.

    When ``data`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the previous close is taken within each group,
    so one ticker's history cannot leak into another's. The returned series
    preserves the input index and row order and is expressed in the input price
    units.
    """
    validate_required_columns(data, required_columns=_REQUIRED_PRICE_COLUMNS)
    return apply_by_ticker(data, _true_range)


def atr(data: pd.DataFrame, *, length: int = 14) -> pd.Series:
    """Calculate the Average True Range for one or many tickers.

    ATR is Wilder's smoothed moving average of :func:`true_range` over ``length``
    rows, so the first ``length - 1`` rows of each ticker remain missing until the
    window is full. ``data`` must contain ``high``, ``low``, and ``close`` columns
    in chronological order.

    When ``data`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the calculation is isolated within each group.
    ATR is expressed in the input price units and is not comparable across
    tickers; it is provided as a standalone indicator for direct use, such as in
    the frontend application, and is not part of :func:`add_volatility_features`.
    """
    validate_length(length)
    validate_required_columns(data, required_columns=_REQUIRED_PRICE_COLUMNS)
    return apply_by_ticker(data, lambda group: _atr(group, length=length))


def atr_percent(data: pd.DataFrame, *, length: int = 14) -> pd.Series:
    """Calculate ATR as a percentage of the closing price for one or many tickers.

    ATR percent divides :func:`atr` by the current close and scales the result to
    percentage points, producing a scale-invariant volatility measure that is
    comparable across tickers. ``data`` must contain ``high``, ``low``, and
    ``close`` columns in chronological order.

    When ``data`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the calculation is isolated within each group.
    """
    validate_length(length)
    validate_required_columns(data, required_columns=_REQUIRED_PRICE_COLUMNS)
    return apply_by_ticker(data, lambda group: _atr_percent(group, length=length))


def _true_range(data: pd.DataFrame) -> pd.Series:
    previous_close = data.loc[:, "close"].shift(1)
    ranges = pd.DataFrame(
        {
            "high_low": data.loc[:, "high"] - data.loc[:, "low"],
            "high_close": (data.loc[:, "high"] - previous_close).abs(),
            "low_close": (data.loc[:, "low"] - previous_close).abs(),
        },
        index=data.index,
    )
    return ranges.max(axis=1).rename("true_range")


def _atr(data: pd.DataFrame, *, length: int) -> pd.Series:
    return wilder_moving_average(_true_range(data), length=length).rename("atr")


def _atr_percent(data: pd.DataFrame, *, length: int) -> pd.Series:
    atr_values = _atr(data, length=length)
    return (100 * safe_divide(atr_values, data.loc[:, "close"])).rename("atr_percent")
