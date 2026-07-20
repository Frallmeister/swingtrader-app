"""Volatility indicators: True Range, ATR, and Bollinger Bands.

True Range and ATR need the intraday extremes together, so they consume a
dataframe with ``high``, ``low``, and ``close`` columns. The Bollinger indicators
operate on a single ordered series so they can also be applied to other signals,
such as an RSI line. Each indicator accepts either a single ordered instrument or
a multi-instrument input carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels, in which case the calculation is isolated within
each provider/ticker group and the input row order is preserved. Each function
returns new index-aligned outputs and does not mutate its input.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._smoothing import wilder_moving_average
from swingtrader.indicators._validation import validate_length, validate_num_std


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
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(data, _true_range)


def atr(data: pd.DataFrame, *, length: int = 14) -> pd.Series:
    """Calculate the Average True Range for one or many tickers.

    ATR is Wilder's smoothed moving average of :func:`true_range` over ``length``
    rows, so the first ``length - 1`` rows of each ticker remain missing until the
    window is full. ``data`` must contain ``high``, ``low``, and ``close`` columns
    in chronological order. The smoothing is the recursive form seeded from the
    first True Range rather than the canonical simple average of the first
    ``length`` True Ranges, so early ATR values differ slightly from a canonical
    implementation before converging (see
    :func:`swingtrader.indicators._smoothing.wilder_moving_average`).

    When ``data`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the calculation is isolated within each group.
    ATR is expressed in the input price units and is not comparable across
    tickers; it is provided as a standalone indicator for direct use, such as in
    the frontend application.
    """
    validate_length(length)
    validate_required_columns(data, required_columns={"high", "low", "close"})
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
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(data, lambda group: _atr_percent(group, length=length))


def bollinger_bands(
    values: pd.Series,
    *,
    length: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Calculate Bollinger Bands for one or many tickers.

    Returns a dataframe with ``bollinger_middle``, ``bollinger_upper``, and
    ``bollinger_lower`` columns. The middle band is the ``length``-row simple
    moving average of ``values``, and the upper and lower bands sit ``num_std``
    rolling standard deviations above and below it. The bands are expressed in the
    input units of ``values``.

    ``values`` is a single ordered series, such as adjusted close or any other
    signal (for example an RSI line), rather than a price dataframe. When it
    carries the canonical ``provider``, ``ticker``, and ``trading_date`` index
    levels the calculation is isolated within each group. The first
    ``length - 1`` rows of each series remain missing until the window is full.

    The rolling standard deviation is the population standard deviation
    (``ddof=0``), matching John Bollinger's original definition and most charting
    platforms; libraries that use the sample standard deviation (``ddof=1``)
    produce slightly wider bands for the same ``length``.
    """
    validate_length(length)
    validate_num_std(num_std)
    return apply_by_ticker(
        values, lambda group: _bollinger_bands(group, length=length, num_std=num_std)
    )


def bollinger_bandwidth(
    values: pd.Series,
    *,
    length: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Calculate Bollinger Bandwidth for one or many tickers.

    Bollinger Bandwidth is the distance between the upper and lower
    :func:`bollinger_bands` divided by the middle band, giving a scale-invariant
    measure of relative band width that is comparable across tickers and signals.
    ``values`` is a single ordered series. When it carries the canonical
    ``provider``, ``ticker``, and ``trading_date`` index levels the calculation is
    isolated within each group.
    """
    validate_length(length)
    validate_num_std(num_std)
    return apply_by_ticker(
        values, lambda group: _bollinger_bandwidth(group, length=length, num_std=num_std)
    )


def bollinger_percent_b(
    values: pd.Series,
    *,
    length: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Calculate the Bollinger %B value for one or many tickers.

    %B locates ``values`` within its :func:`bollinger_bands` as the fraction of
    the distance from the lower band to the upper band: 0 marks the lower band, 1
    the upper band, and values outside ``[0, 1]`` fall beyond the bands.
    ``values`` is a single ordered series. When it carries the canonical
    ``provider``, ``ticker``, and ``trading_date`` index levels the calculation is
    isolated within each group.
    """
    validate_length(length)
    validate_num_std(num_std)
    return apply_by_ticker(
        values, lambda group: _bollinger_percent_b(group, length=length, num_std=num_std)
    )


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


def _bollinger_bands(values: pd.Series, *, length: int, num_std: float) -> pd.DataFrame:
    rolling = values.rolling(window=length, min_periods=length)
    middle = rolling.mean()
    offset = num_std * rolling.std(ddof=0)
    return pd.DataFrame(
        {
            "bollinger_middle": middle,
            "bollinger_upper": middle + offset,
            "bollinger_lower": middle - offset,
        },
        index=values.index,
    )


def _bollinger_bandwidth(values: pd.Series, *, length: int, num_std: float) -> pd.Series:
    bands = _bollinger_bands(values, length=length, num_std=num_std)
    width = bands.loc[:, "bollinger_upper"] - bands.loc[:, "bollinger_lower"]
    return safe_divide(width, bands.loc[:, "bollinger_middle"]).rename("bollinger_bandwidth")


def _bollinger_percent_b(values: pd.Series, *, length: int, num_std: float) -> pd.Series:
    bands = _bollinger_bands(values, length=length, num_std=num_std)
    position = values - bands.loc[:, "bollinger_lower"]
    width = bands.loc[:, "bollinger_upper"] - bands.loc[:, "bollinger_lower"]
    return safe_divide(position, width).rename("bollinger_percent_b")
