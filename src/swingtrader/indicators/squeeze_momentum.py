"""LazyBear Squeeze Momentum indicator.

This module ports the open-source "Squeeze Momentum Indicator [LazyBear]"
published on TradingView. It combines Bollinger Bands, Keltner Channels, and a
linear-regression momentum histogram, returning several related outputs together,
so it lives in its own module. The indicator-specific linear-regression helper
remains private to this module.

The indicator consumes a dataframe with ``high``, ``low``, and ``close`` columns
and computes True Range and ATR internally. It accepts either one instrument or a
multi-instrument input carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels, in which case the calculation is isolated within
each provider/ticker group and the input row order is preserved. The function
returns a new dataframe and does not mutate its input.
"""

import numpy as np
import pandas as pd

from swingtrader.core.numerical import consecutive_true_count, safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._validation import (
    validate_length,
    validate_multiplier,
)
from swingtrader.indicators.moving_averages import sma
from swingtrader.indicators.volatility import _atr, _true_range


def lazybear_squeeze_momentum(
    data: pd.DataFrame,
    *,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    atr_length: int = 14,
) -> pd.DataFrame:
    """Calculate LazyBear's Squeeze Momentum indicator for one or many tickers.

    This is a pandas port of the open-source "Squeeze Momentum Indicator
    [LazyBear]" published on TradingView, itself a derivative of John Carter's TTM
    Squeeze. The squeeze compares Bollinger Bands against Keltner Channels: a
    squeeze is on while the Bollinger Bands sit inside the Keltner Channels (low
    volatility, a coiled market) and off once they expand back outside them. A
    separate linear-regression histogram measures the momentum building up while
    the market is squeezed.

    Returns a dataframe with the following columns:

    - ``squeeze_on``: nullable boolean, true while both Bollinger Bands sit inside
      the Keltner Channels;
    - ``squeeze_off``: nullable boolean, true while both Bollinger Bands sit
      outside the Keltner Channels;
    - ``squeeze_released``: nullable boolean, true on the first row after a
      ``squeeze_on`` row that is no longer squeezed, marking the bar the squeeze
      fires;
    - ``squeeze_width_ratio``: the Bollinger-band width divided by the
      Keltner-channel width, a scale-invariant measure of how compressed the bands
      are, where a value below one means the Bollinger Bands are inside the
      channels;
    - ``squeeze_momentum``: the raw linear-regression momentum histogram in the
      input price units;
    - ``squeeze_momentum_atr``: ``squeeze_momentum`` divided by ATR, a
      scale-invariant momentum measure comparable across tickers;
    - ``squeeze_momentum_atr_change``: the row-over-row change in
      ``squeeze_momentum_atr``, capturing whether momentum is building or fading;
    - ``squeeze_duration``: the number of consecutive rows the current squeeze has
      been on, resetting to zero while it is off;
    - ``squeeze_release_duration``: on each ``squeeze_released`` row, how many rows
      the squeeze that just fired had lasted.

    The Bollinger Bands are the ``bb_length``-row simple moving average of
    ``close`` plus and minus ``bb_mult`` population standard deviations. The
    Keltner Channels are the ``kc_length``-row simple moving average of ``close``
    plus and minus ``kc_mult`` times the ``kc_length``-row average True Range. The
    momentum histogram is a rolling linear regression of ``close`` detrended
    against the midpoint of its recent high/low range and moving average, and it
    is normalised by the ``atr_length``-row ATR. True Range and ATR are computed
    internally from ``high``, ``low``, and ``close``, so the caller does not supply
    them, matching the stochastic and Money Flow Index indicators.

    ``data`` must contain ``high``, ``low``, and ``close`` columns in
    chronological order. When ``data`` carries the canonical ``provider``,
    ``ticker``, and ``trading_date`` index levels the calculation is isolated
    within each group, so one ticker's history cannot leak into another's, and the
    original index and row order are preserved. Warm-up rows remain missing until
    every rolling and smoothing window is full. The squeeze state becomes defined
    once the band windows fill, while the momentum histogram warms up later
    because its detrended input is itself a ``kc_length`` calculation that the
    linear regression then windows again.

    Notes
    -----
    Source: https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
    """
    validate_length(bb_length)
    validate_length(kc_length)
    validate_length(atr_length)
    validate_multiplier(bb_mult)
    validate_multiplier(kc_mult)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _lazybear_squeeze_momentum(
            group,
            bb_length=bb_length,
            bb_mult=bb_mult,
            kc_length=kc_length,
            kc_mult=kc_mult,
            atr_length=atr_length,
        ),
    )


def _lazybear_squeeze_momentum(
    data: pd.DataFrame,
    *,
    bb_length: int,
    bb_mult: float,
    kc_length: int,
    kc_mult: float,
    atr_length: int,
) -> pd.DataFrame:
    # Port of the open-source Squeeze Momentum Indicator [LazyBear]:
    # https://www.tradingview.com/script/nqQ1DT5a-Squeeze-Momentum-Indicator-LazyBear/
    close = data["close"]
    high = data["high"]
    low = data["low"]
    true_range_ = _true_range(data)
    atr_ = _atr(data, length=atr_length)

    # Bollinger Bands
    bb_basis = sma(close, length=bb_length)
    bb_deviation = (
        bb_mult  # The original LazyBear script uses kc_mult here, most likely a mistake.
        * close.rolling(
            window=bb_length,
            min_periods=bb_length,
        ).std(ddof=0)
    )

    upper_bb = bb_basis + bb_deviation
    lower_bb = bb_basis - bb_deviation

    # Keltner Channels
    kc_basis = sma(close, length=kc_length)
    range_ma = sma(true_range_, length=kc_length)

    upper_kc = kc_basis + kc_mult * range_ma
    lower_kc = kc_basis - kc_mult * range_ma

    squeeze_ready = (
        pd.concat(
            [upper_bb, lower_bb, upper_kc, lower_kc],
            axis=1,
        )
        .notna()
        .all(axis=1)
    )

    squeeze_on = (
        ((lower_bb > lower_kc) & (upper_bb < upper_kc)).astype("boolean").where(squeeze_ready)
    )

    squeeze_off = (
        ((lower_bb < lower_kc) & (upper_bb > upper_kc)).astype("boolean").where(squeeze_ready)
    )

    squeeze_width_ratio = safe_divide(upper_bb - lower_bb, upper_kc - lower_kc)
    squeeze_released = squeeze_on.shift(1, fill_value=False) & squeeze_on.eq(False)
    squeeze_duration = consecutive_true_count(squeeze_on)
    squeeze_release_duration = squeeze_duration.shift(1).where(squeeze_released)

    # Calculate momentum
    highest_high = high.rolling(window=kc_length, min_periods=kc_length).max()
    lowest_low = low.rolling(window=kc_length, min_periods=kc_length).min()

    range_midpoint = (highest_high + lowest_low) / 2.0
    reference_level = (range_midpoint + kc_basis) / 2.0
    detrended_close = close - reference_level

    momentum = _linreg(detrended_close, length=kc_length, offset=0)
    momentum_atr = safe_divide(momentum, atr_)
    momentum_atr_change = momentum_atr.diff()

    return pd.DataFrame(
        {
            "squeeze_on": squeeze_on,
            "squeeze_off": squeeze_off,
            "squeeze_released": squeeze_released,
            "squeeze_width_ratio": squeeze_width_ratio,
            "squeeze_momentum": momentum,
            "squeeze_momentum_atr": momentum_atr,
            "squeeze_momentum_atr_change": momentum_atr_change,
            "squeeze_duration": squeeze_duration,
            "squeeze_release_duration": squeeze_release_duration,
        },
        index=data.index,
    )


def _linreg(
    values: pd.Series,
    *,
    length: int,
    offset: int = 0,
) -> pd.Series:
    """Calculate TradingView-style rolling linear regression.

    Fits an ordinary least-squares line to each rolling window and returns
    the fitted value at position ``length - 1 - offset``.

    Parameters
    ----------
    values
        Input series ordered from oldest to newest.
    length
        Number of observations in each regression window.
    offset
        Offset from the newest observation. An offset of zero evaluates
        the fitted line at the newest position.

    Returns
    -------
    pd.Series
        Rolling fitted values with the same index as ``values``.
    """
    if length < 1:
        raise ValueError("length must be at least 1")

    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_centered = x - x_mean
    denominator = np.dot(x_centered, x_centered)

    evaluation_position = length - 1 - offset

    # For length=1, every regression consists of one constant observation.
    if denominator == 0:
        return values.rolling(
            window=length,
            min_periods=length,
        ).mean()

    # The fitted value can be expressed directly as a weighted sum of y.
    # This avoids running np.polyfit for every rolling window.
    weights = (
        np.full(length, 1.0 / length) + (evaluation_position - x_mean) * x_centered / denominator
    )

    return values.rolling(
        window=length,
        min_periods=length,
    ).apply(
        lambda window: np.dot(window, weights),
        raw=True,
    )
