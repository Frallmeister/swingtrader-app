"""Momentum oscillators: RSI and the stochastic oscillator.

RSI operates on a single ordered series so the caller chooses the source. The
stochastic oscillator needs the intraday extremes together, so it consumes a
dataframe with ``high``, ``low``, and ``close`` columns. Both accept either a
single instrument or a multi-instrument input carrying the canonical
``provider``, ``ticker``, and ``trading_date`` index levels, in which case the
calculation is isolated within each provider/ticker group and the input row order
is preserved. Neither function mutates its input.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._smoothing import wilder_moving_average
from swingtrader.indicators._validation import validate_length


def rsi(
    values: pd.Series,
    *,
    length: int = 14,
) -> pd.Series:
    """Calculate Wilder's Relative Strength Index for one or many tickers.

    RSI is a bounded ``[0, 100]`` momentum oscillator built from the average gain
    and average loss over ``length`` rows, each smoothed with Wilder's recursive
    moving average. It is calculated as ``100 * avg_gain / (avg_gain + avg_loss)``,
    so a window with no losses returns 100 and a window with no gains returns 0. A
    fully flat window has neither gains nor losses and is left missing.

    ``values`` is a single ordered series, so the caller chooses the source, such
    as close, adjusted close, or an OHLC average. When it carries the canonical
    ``provider``, ``ticker``, and ``trading_date`` index levels the calculation is
    isolated within each group. The first ``length`` rows of each series remain
    missing until the smoothing window is full.

    The smoothing is the recursive form seeded from the first change rather than
    the canonical Wilder definition that seeds from the simple average of the
    first ``length`` changes, so early RSI values differ slightly from a canonical
    implementation before converging (see
    :func:`swingtrader.indicators._smoothing.wilder_moving_average`).
    """
    validate_length(length)
    return apply_by_ticker(values, lambda group: _rsi(group, length=length))


def stochastic_oscillator(
    data: pd.DataFrame,
    *,
    k_length: int = 14,
    k_smoothing: int = 3,
    d_length: int = 3,
) -> pd.DataFrame:
    """Calculate the stochastic oscillator for one or many tickers.

    Returns a dataframe with ``stochastic_k`` and ``stochastic_d`` columns. The
    raw %K locates the close within its recent range as
    ``100 * (close - lowest_low) / (highest_high - lowest_low)`` over ``k_length``
    rows, where ``lowest_low`` and ``highest_high`` are the rolling minimum low
    and maximum high over the same window. ``stochastic_k`` is that raw %K
    smoothed with a simple moving average over ``k_smoothing`` rows, and
    ``stochastic_d`` is a further simple moving average of ``stochastic_k`` over
    ``d_length`` rows. Passing ``k_smoothing=1`` yields the fast stochastic; the
    conventional 14/3/3 defaults yield the slow stochastic. Both series are
    bounded to ``[0, 100]``.

    ``data`` must contain ``high``, ``low``, and ``close`` columns in
    chronological order, because the oscillator needs the intraday extremes and
    the close together. When ``data`` carries the canonical ``provider``,
    ``ticker``, and ``trading_date`` index levels the calculation is isolated
    within each group, so one ticker's history cannot leak into another's, and
    the original index and row order are preserved. A window whose highest high
    equals its lowest low has no range and is left missing, and the warm-up rows
    of each series remain missing until every rolling window is full.
    """
    validate_length(k_length)
    validate_length(k_smoothing)
    validate_length(d_length)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(
        data,
        lambda group: _stochastic_oscillator(
            group,
            k_length=k_length,
            k_smoothing=k_smoothing,
            d_length=d_length,
        ),
    )


def _rsi(values: pd.Series, *, length: int) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0)
    loss = delta.mul(-1).clip(lower=0)
    avg_gain = wilder_moving_average(gain, length=length)
    avg_loss = wilder_moving_average(loss, length=length)
    return (100 * safe_divide(avg_gain, avg_gain + avg_loss)).rename("rsi")


def _stochastic_oscillator(
    data: pd.DataFrame,
    *,
    k_length: int,
    k_smoothing: int,
    d_length: int,
) -> pd.DataFrame:
    high = data.loc[:, "high"]
    low = data.loc[:, "low"]
    close = data.loc[:, "close"]

    lowest_low = low.rolling(window=k_length, min_periods=k_length).min()
    highest_high = high.rolling(window=k_length, min_periods=k_length).max()
    raw_k = 100 * safe_divide(close - lowest_low, highest_high - lowest_low)
    stochastic_k = raw_k.rolling(window=k_smoothing, min_periods=k_smoothing).mean()
    stochastic_d = stochastic_k.rolling(window=d_length, min_periods=d_length).mean()

    return pd.DataFrame(
        {
            "stochastic_k": stochastic_k,
            "stochastic_d": stochastic_d,
        },
        index=data.index,
    )
