"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time trend features from ordered daily
price observations. Calculations are isolated by provider/ticker groups and
leave warm-up periods as missing values until each rolling or exponential window
has enough prior observations.

Numerical indicators accept either one ordered series for a single ticker or a
multi-ticker series carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels. In the latter case the calculation is applied
independently within each provider/ticker group and the input row order is
preserved. The family orchestrator returns a copy of the input dataframe with
final model feature columns appended. The module currently implements
moving-average trend features and is intended to later host directional
indicators such as ADX, +DI, and -DI.
"""

import pandas as pd

from swingtrader.data.features._numerical import (
    exponential_moving_average,
    safe_divide,
)
from swingtrader.data.features._validation import (
    apply_by_ticker,
    validate_length,
    validate_market_price_index,
    validate_required_columns,
)


def add_trend_features(
    data: pd.DataFrame,
    *,
    fast_slow_lengths: tuple[int, int] = (20, 50),
) -> pd.DataFrame:
    """Return a copy of data with the default trend feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus an
    ``adjusted_close`` column. The index must be unique and sorted. The returned
    dataframe preserves the input rows and appends the final moving-average ratio
    feature columns.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"adjusted_close"})

    fast, slow = fast_slow_lengths
    validate_length(fast)
    validate_length(slow)
    if fast >= slow:
        raise ValueError(
            f"The fast length must be lower than the slow length; got fast={fast!r}, slow={slow!r}"
        )

    data = data.copy()
    adjusted_close_by_ticker = data.loc[:, "adjusted_close"].groupby(
        level=["provider", "ticker"],
        sort=False,
    )

    sma_fast = adjusted_close_by_ticker.transform(lambda values: _sma(values, length=fast))
    sma_slow = adjusted_close_by_ticker.transform(lambda values: _sma(values, length=slow))
    ema_fast = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=fast))
    ema_slow = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=slow))

    data["sma_fast_to_sma_slow"] = safe_divide(sma_fast, sma_slow).sub(1)
    data["ema_fast_to_ema_slow"] = safe_divide(ema_fast, ema_slow).sub(1)
    data["ema_fast_to_sma_fast"] = safe_divide(ema_fast, sma_fast).sub(1)

    return data


def sma(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate a simple moving average for one or many tickers.

    ``values`` must contain observations in chronological order. When ``values``
    carries the canonical ``provider``, ``ticker``, and ``trading_date`` index
    levels the average is calculated independently within each group. The
    returned series preserves the input index, with the first ``length - 1``
    observations of each series left missing until the rolling window is full.
    """
    validate_length(length)
    return apply_by_ticker(values, lambda group: _sma(group, length=length))


def ema(
    values: pd.Series,
    *,
    length: int,
) -> pd.Series:
    """Calculate an exponential moving average for one or many tickers.

    ``values`` must contain observations in chronological order. When ``values``
    carries the canonical ``provider``, ``ticker``, and ``trading_date`` index
    levels the average is calculated independently within each group. The
    returned series preserves the input index. EMA uses pandas ``ewm`` with
    ``span=length``, ``adjust=False``, and ``min_periods=length``.
    """
    validate_length(length)
    return apply_by_ticker(values, lambda group: _ema(group, length=length))


def _sma(values: pd.Series, *, length: int) -> pd.Series:
    return values.rolling(window=length, min_periods=length).mean()


def _ema(values: pd.Series, *, length: int) -> pd.Series:
    return exponential_moving_average(values, length=length)
