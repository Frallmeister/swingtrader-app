"""Simple and exponential moving-average indicators.

These reusable moving averages accept either one ordered series for a single
instrument or a multi-instrument series carrying the canonical ``provider``,
``ticker``, and ``trading_date`` index levels. In the latter case the calculation
is applied independently within each provider/ticker group and the input row
order is preserved. Each function returns one index-aligned series and does not
mutate its input.
"""

import pandas as pd

from swingtrader.data.market_frame import apply_by_ticker
from swingtrader.indicators._smoothing import exponential_moving_average
from swingtrader.indicators._validation import validate_length


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
