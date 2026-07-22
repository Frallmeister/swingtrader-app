"""Candlestick geometry, range context, and local pattern indicators.

The indicators in this module combine continuous OHLC descriptions with a small
set of direct local candle relations. They preserve body, wick, close-location,
gap, range, containment, engulfing, and rejection information without adding a
large catalogue of thresholded textbook patterns.

Each public function accepts either one chronologically ordered instrument or a
canonical multi-instrument market frame carrying the ``provider``, ``ticker``,
and ``trading_date`` index levels. Multi-instrument calculations are isolated by
provider and ticker, preserve the input index and row order, and never mutate the
input dataframe.
"""

import pandas as pd

from swingtrader.core.numerical import consecutive_true_count, safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._validation import validate_length
from swingtrader.indicators.volatility import _atr, _true_range


def candle_geometry(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate normalized geometry for each OHLC candle.

    ``data`` must contain ``open``, ``high``, ``low``, and ``close`` columns.
    The returned dataframe contains:

    - ``signed_body_fraction``: ``(close - open) / (high - low)``;
    - ``upper_wick_fraction``: the upper wick divided by the high-low range;
    - ``lower_wick_fraction``: the lower wick divided by the high-low range;
    - ``close_location``: the close's position from low (0) to high (1).

    A positive signed body is bullish and a negative body is bearish. Zero-range
    candles cannot be normalized and therefore produce missing values rather
    than infinities. The function does not impose candle-validity constraints or
    clip outputs; it describes the supplied OHLC values as-is.
    """
    validate_required_columns(data, required_columns={"open", "high", "low", "close"})
    return apply_by_ticker(data, _candle_geometry)


def candle_range_context(
    data: pd.DataFrame,
    *,
    atr_length: int = 14,
    percentile_length: int = 20,
) -> pd.DataFrame:
    """Calculate point-in-time gap, range-expansion, and range-rank indicators.

    ``range_atr`` divides the current True Range by the ATR available at the end
    of the previous row. ``gap_atr`` divides the signed opening gap from the
    previous close by that same prior ATR. Using prior ATR prevents the current
    event from increasing its own normalization denominator.

    ``range_percentile`` is the fraction of the preceding ``percentile_length``
    high-low ranges that are less than or equal to the current high-low range.
    The current row is excluded from the reference window. Values therefore lie
    in ``[0, 1]``, where values near one identify locally wide candles and values
    near zero identify locally narrow candles.

    The first rows of each instrument remain missing until the required ATR and
    percentile histories are available. ``data`` must contain ``open``, ``high``,
    ``low``, and ``close`` columns in chronological order.
    """
    validate_length(atr_length)
    validate_length(percentile_length)
    validate_required_columns(data, required_columns={"open", "high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _candle_range_context(
            group,
            atr_length=atr_length,
            percentile_length=percentile_length,
        ),
    )


def candle_patterns(
    data: pd.DataFrame,
    *,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Calculate local containment, engulfing, and wick-rejection signals.

    ``inside_bar`` and ``outside_bar`` compare the current high-low range with
    the previous candle. Equality at one boundary is allowed, but an unchanged
    range is neither inside nor outside. Comparison outputs remain missing until
    a complete previous candle is available. ``consecutive_inside_bars`` counts
    the current inside-bar streak and resets to zero otherwise.

    ``engulfing_strength`` is the signed amount by which an opposite-direction
    real body exceeds and contains the previous real body, normalized by the ATR
    known on the previous row. Positive values are bullish and negative values
    bearish. ``lower_rejection_strength`` and ``upper_rejection_strength`` scale
    the corresponding wick by the same prior ATR and weight it by where the
    candle closed within its range. Strength outputs remain missing when their
    required prior ATR is unavailable.

    ``data`` must contain ``open``, ``high``, ``low``, and ``close`` columns in
    chronological order.
    """
    validate_length(atr_length)
    validate_required_columns(data, required_columns={"open", "high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _candle_patterns(group, atr_length=atr_length),
    )


def _candle_geometry(data: pd.DataFrame) -> pd.DataFrame:
    open_values = data.loc[:, "open"]
    high_values = data.loc[:, "high"]
    low_values = data.loc[:, "low"]
    close_values = data.loc[:, "close"]

    candle_range = high_values - low_values
    body_edges = pd.concat([open_values, close_values], axis=1)
    upper_body = body_edges.max(axis=1, skipna=False)
    lower_body = body_edges.min(axis=1, skipna=False)

    return pd.DataFrame(
        {
            "signed_body_fraction": safe_divide(close_values - open_values, candle_range),
            "upper_wick_fraction": safe_divide(high_values - upper_body, candle_range),
            "lower_wick_fraction": safe_divide(lower_body - low_values, candle_range),
            "close_location": safe_divide(close_values - low_values, candle_range),
        },
        index=data.index,
    )


def _candle_range_context(
    data: pd.DataFrame,
    *,
    atr_length: int,
    percentile_length: int,
) -> pd.DataFrame:
    prior_atr = _atr(data, length=atr_length).shift(1)
    current_true_range = _true_range(data)
    opening_gap = data.loc[:, "open"] - data.loc[:, "close"].shift(1)
    candle_range = data.loc[:, "high"] - data.loc[:, "low"]

    return pd.DataFrame(
        {
            "range_atr": safe_divide(current_true_range, prior_atr),
            "gap_atr": safe_divide(opening_gap, prior_atr),
            "range_percentile": _prior_range_percentile(
                candle_range,
                length=percentile_length,
            ),
        },
        index=data.index,
    )


def _candle_patterns(data: pd.DataFrame, *, atr_length: int) -> pd.DataFrame:
    open_values = data.loc[:, "open"]
    high_values = data.loc[:, "high"]
    low_values = data.loc[:, "low"]
    close_values = data.loc[:, "close"]

    previous_high = high_values.shift(1)
    previous_low = low_values.shift(1)
    comparable_ranges = (
        high_values.notna() & low_values.notna() & previous_high.notna() & previous_low.notna()
    )

    inside_bar = (
        (
            high_values.le(previous_high)
            & low_values.ge(previous_low)
            & (high_values.lt(previous_high) | low_values.gt(previous_low))
        )
        .astype("boolean")
        .where(comparable_ranges)
    )
    outside_bar = (
        (
            high_values.ge(previous_high)
            & low_values.le(previous_low)
            & (high_values.gt(previous_high) | low_values.lt(previous_low))
        )
        .astype("boolean")
        .where(comparable_ranges)
    )

    body_edges = pd.concat([open_values, close_values], axis=1)
    upper_body = body_edges.max(axis=1, skipna=False)
    lower_body = body_edges.min(axis=1, skipna=False)
    previous_upper_body = upper_body.shift(1)
    previous_lower_body = lower_body.shift(1)
    body = close_values - open_values
    previous_body = body.shift(1)

    comparable_bodies = body.notna() & previous_body.notna()
    opposite_direction = body.mul(previous_body).lt(0)
    contains_previous_body = (
        lower_body.le(previous_lower_body)
        & upper_body.ge(previous_upper_body)
        & (lower_body.lt(previous_lower_body) | upper_body.gt(previous_upper_body))
    )

    prior_atr = _atr(data, length=atr_length).shift(1)
    body_direction = safe_divide(body, body.abs())
    body_excess_atr = safe_divide(body.abs() - previous_body.abs(), prior_atr)
    engulfing = comparable_bodies & opposite_direction & contains_previous_body

    engulfing_strength = body_direction.mul(body_excess_atr).where(engulfing, 0.0)
    engulfing_strength = engulfing_strength.where(comparable_bodies & prior_atr.notna())

    close_location = safe_divide(close_values - low_values, high_values - low_values)
    lower_wick_atr = safe_divide(lower_body - low_values, prior_atr)
    upper_wick_atr = safe_divide(high_values - upper_body, prior_atr)

    return pd.DataFrame(
        {
            "inside_bar": inside_bar,
            "outside_bar": outside_bar,
            "engulfing_strength": engulfing_strength,
            "lower_rejection_strength": lower_wick_atr.mul(close_location),
            "upper_rejection_strength": upper_wick_atr.mul(1.0 - close_location),
            "consecutive_inside_bars": consecutive_true_count(inside_bar),
        },
        index=data.index,
    )


def _prior_range_percentile(values: pd.Series, *, length: int) -> pd.Series:
    """Return the current value's percentile among the previous `length` values."""
    prior_values = pd.concat(
        [values.shift(offset) for offset in range(1, length + 1)],
        axis=1,
    )
    complete_history = prior_values.notna().all(axis=1) & values.notna()
    percentile = prior_values.le(values, axis=0).sum(axis=1).div(length)
    return percentile.where(complete_history).rename("range_percentile")
