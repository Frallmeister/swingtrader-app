"""Volatility feature transformations for daily price histories.

This module builds row-aligned, point-in-time volatility features from ordered
daily price observations. Calculations are isolated by provider/ticker groups
and leave warm-up periods as missing values until each rolling or smoothing
window has enough prior observations.

Numerical indicators operate independently per ticker: when the input carries
the canonical ``provider``, ``ticker``, and ``trading_date`` index levels the
calculation is applied within each group and the input row order is preserved.
True Range and ATR consume several price columns and therefore take a dataframe
with ``high``, ``low``, and ``close`` columns, while the Bollinger indicators
operate on a single ordered series so they can also be applied to other signals,
such as a future RSI. Each indicator returns one index-aligned series, except
:func:`bollinger_bands`, which returns a dataframe of the three bands. The family
orchestrator returns a copy of the input dataframe with the final model feature
columns appended. The module currently implements True Range, ATR-based, and
Bollinger-band features.
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


def add_volatility_features(
    data: pd.DataFrame,
    *,
    atr_length: int = 14,
    bollinger_length: int = 20,
    bollinger_num_std: float = 2.0,
) -> pd.DataFrame:
    """Return a copy of data with the default volatility feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus
    ``high``, ``low``, ``close``, and ``adjusted_close`` columns. The index must
    be unique and sorted. The returned dataframe preserves the input rows and
    appends the final ``atr_percent``, ``bollinger_bandwidth``, and
    ``bollinger_percent_b`` feature columns.

    ATR is calculated from raw ``high``, ``low``, and ``close`` because True Range
    needs the intraday extremes together. The Bollinger features are calculated
    from ``adjusted_close`` so their rolling mean and standard deviation are not
    distorted by split and dividend discontinuities in the raw close.

    Raw True Range, ATR, and the Bollinger bands themselves are expressed in the
    input price units and are not comparable across tickers, so the orchestrator
    only appends the scale-invariant columns. Use :func:`true_range`,
    :func:`atr`, and :func:`bollinger_bands` directly when absolute price-unit
    values are required.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"high", "low", "close", "adjusted_close"})
    validate_length(atr_length)
    validate_length(bollinger_length)
    _validate_num_std(bollinger_num_std)

    data = data.copy()
    data["atr_percent"] = atr_percent(data, length=atr_length)

    adjusted_close = data.loc[:, "adjusted_close"]
    data["bollinger_bandwidth"] = bollinger_bandwidth(
        adjusted_close, length=bollinger_length, num_std=bollinger_num_std
    )
    data["bollinger_percent_b"] = bollinger_percent_b(
        adjusted_close, length=bollinger_length, num_std=bollinger_num_std
    )
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
    :func:`swingtrader.data.features._numerical.wilder_moving_average`).

    When ``data`` carries the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels the calculation is isolated within each group.
    ATR is expressed in the input price units and is not comparable across
    tickers; it is provided as a standalone indicator for direct use, such as in
    the frontend application, and is not part of :func:`add_volatility_features`.
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
    signal (for example a future RSI), rather than a price dataframe. When it
    carries the canonical ``provider``, ``ticker``, and ``trading_date`` index
    levels the calculation is isolated within each group. The first
    ``length - 1`` rows of each series remain missing until the window is full.

    The rolling standard deviation is the population standard deviation
    (``ddof=0``), matching John Bollinger's original definition and most charting
    platforms; libraries that use the sample standard deviation (``ddof=1``)
    produce slightly wider bands for the same ``length``.

    The bands are exposed as a standalone indicator for exploratory analysis and
    frontend charts. They are not part of :func:`add_volatility_features`, which
    instead appends the scale-invariant ``bollinger_bandwidth`` and
    ``bollinger_percent_b`` features.
    """
    validate_length(length)
    _validate_num_std(num_std)
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
    _validate_num_std(num_std)
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
    _validate_num_std(num_std)
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


def _validate_num_std(num_std: float) -> None:
    if isinstance(num_std, bool) or not isinstance(num_std, int | float) or num_std <= 0:
        raise ValueError(f"num_std must be a positive number; got {num_std!r}")
