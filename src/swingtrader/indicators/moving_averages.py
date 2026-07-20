"""Simple and exponential moving-average indicators.

These reusable moving averages accept either one ordered series for a single
instrument or a multi-instrument series carrying the canonical ``provider``,
``ticker``, and ``trading_date`` index levels. In the latter case the calculation
is applied independently within each provider/ticker group and the input row
order is preserved. Each function returns one index-aligned series and does not
mutate its input.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
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


def rolling_vwap(data: pd.DataFrame, *, length: int) -> pd.Series:
    """Calculate the rolling volume-weighted average price.

    The representative price for each observation is its typical price,
    calculated as ``(high + low + close) / 3``. The rolling VWAP is then the
    sum of typical price multiplied by volume divided by the corresponding
    rolling sum of volume.

    Calculations are performed independently for each provider/ticker group.
    A value is returned only after a complete rolling window is available.

    Args:
        data: Ordered daily market data containing ``high``, ``low``, ``close``,
            and ``volume`` columns, together with the identifiers required to
            group observations by ticker.
        length: Number of observations in the rolling window. Must be positive.

    Returns:
        A row-aligned series containing the rolling VWAP. The first
        ``length - 1`` observations in each provider/ticker group are missing.
        Values are also missing when the rolling volume denominator is zero or
        invalid.

    Raises:
        ValueError: If ``length`` is not positive.
        KeyError: If a required column is missing.
    """
    validate_length(length)
    validate_required_columns(data, required_columns={"high", "low", "close", "volume"})
    return apply_by_ticker(data, lambda group: _rolling_vwap(group, length=length))


def _sma(values: pd.Series, *, length: int) -> pd.Series:
    return values.rolling(window=length, min_periods=length).mean()


def _ema(values: pd.Series, *, length: int) -> pd.Series:
    return exponential_moving_average(values, length=length)


def _rolling_vwap(data: pd.DataFrame, length: int) -> pd.Series:
    typical_price = data[["high", "low", "close"]].mean(axis=1)
    volume = data["volume"]
    price_volume = typical_price * volume
    return safe_divide(
        price_volume.rolling(length, min_periods=length).sum(),
        volume.rolling(length, min_periods=length).sum(),
    )
