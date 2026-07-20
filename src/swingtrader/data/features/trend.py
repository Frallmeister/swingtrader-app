"""Trend-following feature transformations for adjusted-close price histories.

This module builds row-aligned, point-in-time trend features from ordered daily
price observations. It composes reusable indicators from
:mod:`swingtrader.indicators` into model-facing feature columns, deciding which
source columns each indicator uses, how the moving averages are combined into
normalized ratios, and what the final feature columns are named. Calculations are
isolated by provider/ticker groups and leave warm-up periods as missing values
until each underlying window has enough prior observations. The family
orchestrator returns a copy of the input dataframe with the final model feature
columns appended and does not mutate its input. The module currently implements
moving-average trend features and the ADX directional-movement system.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.indicators.directional_movement import adx
from swingtrader.indicators.moving_averages import ema, sma


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

    adx_block = adx(data.loc[:, ["high", "low", "close"]], length=adx_length)
    data[adx_block.columns] = adx_block

    return data
