"""Wilder's directional-movement system (ADX, +DI, -DI).

The directional-movement system needs the intraday extremes together, so it
consumes a dataframe with ``high``, ``low``, and ``close`` columns rather than a
single series. It accepts either one ordered dataframe for a single instrument or
a multi-instrument dataframe carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels, in which case the calculation is isolated within
each provider/ticker group and the input row order is preserved. The function
returns a new dataframe and does not mutate its input.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import apply_by_ticker, validate_required_columns
from swingtrader.indicators._smoothing import wilder_moving_average
from swingtrader.indicators._validation import validate_length


def adx(
    data: pd.DataFrame,
    *,
    length: int = 14,
) -> pd.DataFrame:
    """Calculate Wilder's directional-movement system for one or many tickers.

    Returns a dataframe with ``adx``, ``plus_di``, and ``minus_di`` columns. The
    positive and negative directional indicators measure the share of smoothed
    True Range attributable to upward and downward directional movement over
    ``length`` rows, and ADX is Wilder's smoothed moving average of the
    directional index ``DX`` over the same ``length``. All three are bounded to
    ``[0, 100]``: ``plus_di`` and ``minus_di`` gauge trend direction while ``adx``
    gauges trend strength regardless of direction.

    ``data`` must contain ``high``, ``low``, and ``close`` columns in
    chronological order, because the directional-movement system needs the
    intraday extremes together. When ``data`` carries the canonical ``provider``,
    ``ticker``, and ``trading_date`` index levels the calculation is isolated
    within each group, so one ticker's history cannot leak into another's, and
    the original index and row order are preserved.

    Both the directional indicators and ADX use Wilder's recursive smoothing
    seeded from the first observation rather than the canonical Wilder definition
    that seeds from the simple average of the first ``length`` observations, so
    early values differ slightly from a canonical implementation before
    converging (see
    :func:`swingtrader.indicators._smoothing.wilder_moving_average`). Because ADX
    smooths ``DX`` a second time, its warm-up spans roughly ``2 * length`` rows
    before values become populated.
    """
    validate_length(length)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(data, lambda group: _adx(group, length=length))


def _adx(data: pd.DataFrame, *, length: int) -> pd.DataFrame:
    high = data.loc[:, "high"]
    low = data.loc[:, "low"]
    close = data.loc[:, "close"]

    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    smoothed_true_range = wilder_moving_average(true_range, length=length)
    plus_di = 100 * safe_divide(wilder_moving_average(plus_dm, length=length), smoothed_true_range)
    minus_di = 100 * safe_divide(
        wilder_moving_average(minus_dm, length=length), smoothed_true_range
    )

    directional_index = 100 * safe_divide((plus_di - minus_di).abs(), plus_di + minus_di)
    adx_values = wilder_moving_average(directional_index, length=length)

    return pd.DataFrame(
        {
            "adx": adx_values,
            "plus_di": plus_di,
            "minus_di": minus_di,
        },
        index=data.index,
    )
