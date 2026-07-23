"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time trend features from ordered daily price observations.
It composes reusable indicators from :mod:`swingtrader.indicators` into model-facing feature
columns, deciding which source columns each indicator uses, how the moving averages are combined
into normalized ratios, and what the final feature columns are named. Calculations are isolated by
provider/ticker groups and leave warm-up periods as missing values until each underlying window has
enough prior observations. The family orchestrator returns a copy of the input dataframe with the
final model feature columns appended and does not mutate its input. The module currently implements
moving-average trend features, the ADX directional-movement system, and rolling VWAP displacement
features.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.features._price_adjustment import (
    adjustment_consistent_price_frame,
)
from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)
from swingtrader.indicators import bollinger_percent_b
from swingtrader.indicators.directional_movement import adx
from swingtrader.indicators.moving_averages import ema, rolling_vwap, sma

_TREND_FEATURE_COLUMNS = (
    "ema_fast_to_ema_mid",
    "ema_mid_to_ema_slow",
    "ema_mid_to_sma_mid",
    "close_to_ema_fast",
    "close_to_ema_mid",
    "close_to_ema_slow",
    "adx",
    "plus_di",
    "minus_di",
    "vwap_distance",
    "vwap_distance_percent_b",
)


def add_trend_features(
    data: pd.DataFrame,
    *,
    ma_lengths: tuple[int, int, int] = (10, 20, 50),
    adx_length: int = 14,
    vwap_length: int = 20,
    vwap_bollinger_length: int = 20,
    vwap_bollinger_num_std: float = 2.0,
) -> pd.DataFrame:
    """Return a copy of data with the default trend feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus
    ``high``, ``low``, ``close``, ``volume``, and ``adjusted_close`` columns.
    The index must be unique and sorted. The returned dataframe preserves the
    input rows and appends the final moving-average ratio, directional-movement,
    and rolling-VWAP feature columns.

    The moving-average ratios are calculated from ``adjusted_close``. ADX and
    rolling VWAP receive ``high``, ``low``, and ``close`` transformed onto the
    same adjusted-close scale. This preserves each session's candle geometry
    while preventing splits and dividend adjustments from appearing as market
    movement between sessions.

    Rolling VWAP uses adjustment-consistent typical price
    ``(high + low + close) / 3`` together with source ``volume``.
    ``vwap_distance`` is adjusted close divided by rolling VWAP minus one,
    so positive values indicate that close is above VWAP and negative values
    indicate that it is below VWAP. ``vwap_distance_percent_b`` locates that
    distance within its own rolling Bollinger Bands, indicating how unusual the
    current displacement is relative to its recent history.
    """
    validate_market_price_index(data)
    validate_required_columns(
        data,
        required_columns={"high", "low", "close", "volume", "adjusted_close"},
    )
    validate_new_columns(data, new_columns=_TREND_FEATURE_COLUMNS)

    for length in ma_lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(f"Length must be a positive integer; got {length!r}")

    fast, mid, slow = ma_lengths
    if not fast < mid < slow:
        raise ValueError(
            f"The MA lengths must be in ascending order; got ({fast!r}, {mid!r}, {slow!r})"
        )

    data = data.copy()
    adjusted_close = data.loc[:, "adjusted_close"]
    adjusted_hlc = adjustment_consistent_price_frame(
        data,
        price_columns=("high", "low", "close"),
    )

    sma_mid = sma(adjusted_close, length=mid)
    ema_fast = ema(adjusted_close, length=fast)
    ema_mid = ema(adjusted_close, length=mid)
    ema_slow = ema(adjusted_close, length=slow)

    data["ema_fast_to_ema_mid"] = safe_divide(ema_fast, ema_mid).sub(1)
    data["ema_mid_to_ema_slow"] = safe_divide(ema_mid, ema_slow).sub(1)
    data["ema_mid_to_sma_mid"] = safe_divide(ema_mid, sma_mid).sub(1)
    data["close_to_ema_fast"] = safe_divide(adjusted_close, ema_fast).sub(1)
    data["close_to_ema_mid"] = safe_divide(adjusted_close, ema_mid).sub(1)
    data["close_to_ema_slow"] = safe_divide(adjusted_close, ema_slow).sub(1)

    adx_block = adx(adjusted_hlc, length=adx_length)
    data[adx_block.columns] = adx_block

    adjusted_hlcv = adjusted_hlc.assign(volume=data.loc[:, "volume"])
    vwap_distance = safe_divide(
        adjusted_close,
        rolling_vwap(adjusted_hlcv, length=vwap_length),
    ).sub(1)
    data["vwap_distance"] = vwap_distance
    data["vwap_distance_percent_b"] = bollinger_percent_b(
        vwap_distance,
        length=vwap_bollinger_length,
        num_std=vwap_bollinger_num_std,
    ).rename("vwap_distance_percent_b")
    return data
