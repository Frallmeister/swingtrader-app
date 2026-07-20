"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time trend features from ordered daily
price observations. Calculations are isolated by provider/ticker groups and
leave warm-up periods as missing values until each rolling or exponential window
has enough prior observations.

Numerical indicators accept either one ordered series for a single ticker or a
multi-ticker series carrying the canonical ``provider``, ``ticker``, and
``trading_date`` index levels. In the latter case the calculation is applied
independently within each provider/ticker group and the input row order is
preserved. Indicators return either one series or, for naturally multi-output
indicators, one dataframe. The family orchestrator returns a copy of the input
dataframe with final model feature columns appended. The module currently
implements moving-average trend features and the ADX directional-movement
system.
"""

import pandas as pd

from swingtrader.data.features._numerical import (
    exponential_moving_average,
    safe_divide,
    wilder_moving_average,
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
    ma_lengths: tuple[int, int, int] = (10, 20, 50),
    adx_length: int = 14,
) -> pd.DataFrame:
    """Return a copy of data with the default trend feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus
    ``high``, ``low``, ``close``, and ``adjusted_close`` columns. The index must
    be unique and sorted. The returned dataframe preserves the input rows and
    appends the final moving-average ratio and directional-movement feature
    columns.

    The moving-average ratios are calculated from ``adjusted_close`` so they are
    not distorted by split and dividend discontinuities in the raw close. ADX,
    ``plus_di``, and ``minus_di`` are calculated from the raw ``high``, ``low``,
    and ``close`` because the directional-movement system needs the intraday
    extremes together, matching the ATR calculation in the volatility module.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"high", "low", "close", "adjusted_close"})

    fast, mid, slow = ma_lengths
    validate_length(fast)
    validate_length(mid)
    validate_length(slow)
    if not fast < mid < slow:
        raise ValueError(
            f"The MA lengths must be in ascending order; got ({fast!r}, {mid!r}, {slow!r})"
        )
    validate_length(adx_length)

    data = data.copy()
    adjusted_close_by_ticker = data.loc[:, "adjusted_close"].groupby(
        level=["provider", "ticker"],
        sort=False,
    )

    adjusted_close = data.loc[:, "adjusted_close"]
    sma_mid = adjusted_close_by_ticker.transform(lambda values: _sma(values, length=mid))
    ema_fast = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=fast))
    ema_mid = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=mid))
    ema_slow = adjusted_close_by_ticker.transform(lambda values: _ema(values, length=slow))

    data["ema_fast_to_ema_mid"] = safe_divide(ema_fast, ema_mid).sub(1)
    data["ema_mid_to_ema_slow"] = safe_divide(ema_mid, ema_slow).sub(1)
    data["ema_mid_to_sma_mid"] = safe_divide(ema_mid, sma_mid).sub(1)
    data["close_to_ema_fast"] = safe_divide(adjusted_close, ema_fast).sub(1)
    data["close_to_ema_mid"] = safe_divide(adjusted_close, ema_mid).sub(1)
    data["close_to_ema_slow"] = safe_divide(adjusted_close, ema_slow).sub(1)

    adx_block = adx(data.loc[:, ["high", "low", "close"]], length=adx_length)
    data[adx_block.columns] = adx_block

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
    :func:`swingtrader.data.features._numerical.wilder_moving_average`). Because
    ADX smooths ``DX`` a second time, its warm-up spans roughly ``2 * length``
    rows before values become populated.
    """
    validate_length(length)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    return apply_by_ticker(data, lambda group: _adx(group, length=length))


def _sma(values: pd.Series, *, length: int) -> pd.Series:
    return values.rolling(window=length, min_periods=length).mean()


def _ema(values: pd.Series, *, length: int) -> pd.Series:
    return exponential_moving_average(values, length=length)


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
