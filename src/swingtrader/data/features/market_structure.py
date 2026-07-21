"""Point-in-time market-structure feature transformations.

This module converts confirmed Zig Zag state into normalized, row-aligned
features. Unlike the retrospective :func:`swingtrader.indicators.zigzag` output,
these features update only when each pivot has enough right-side observations to
be confirmed. Intermediate endpoints remain visible until a later confirmed,
more-extreme endpoint replaces them.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import (
    apply_by_ticker,
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.indicators.market_structure import _confirmed_zigzag_state

_ZIGZAG_FEATURE_COLUMNS = [
    "zigzag_last_direction",
    "zigzag_last_swing_return",
    "zigzag_last_swing_bars",
    "zigzag_swing_return_per_bar",
    "zigzag_bars_since_pivot",
    "zigzag_retracement",
]


def add_market_structure_features(
    data: pd.DataFrame,
    *,
    zigzag_deviation: float = 5.0,
    zigzag_pivot_legs: int = 10,
) -> pd.DataFrame:
    """Return a copy of data with the default market-structure features added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date`` and contain ``high``, ``low``,
    and ``close`` columns. The appended features are point-in-time: a Zig Zag
    pivot affects the output only on and after its confirmation row.

    Use :func:`zigzag_features` directly when only the feature block is needed,
    for example for a frontend endpoint that should not calculate every feature
    family.
    """
    result = data.copy()
    feature_block = zigzag_features(
        data,
        deviation=zigzag_deviation,
        pivot_legs=zigzag_pivot_legs,
    )
    result[feature_block.columns] = feature_block
    return result


def zigzag_features(
    data: pd.DataFrame,
    *,
    deviation: float = 5.0,
    pivot_legs: int = 10,
) -> pd.DataFrame:
    """Calculate point-in-time features from the latest confirmed Zig Zag state.

    Returns the following columns:

    - ``zigzag_last_direction``: ``1`` when the latest confirmed endpoint is a
      swing high and ``-1`` for a swing low. Values are missing before the
      first endpoint is confirmed;
    - ``zigzag_last_swing_return``: latest endpoint divided by the preceding
      endpoint minus one;
    - ``zigzag_last_swing_bars``: observations between the latest two pivot rows;
    - ``zigzag_swing_return_per_bar``: geometric mean return per observation over
      the latest retained swing;
    - ``zigzag_bars_since_pivot``: observations from the latest pivot row to the
      current row. Because the feature is emitted on confirmation, its first
      populated value is at least ``pivot_legs // 2``;
    - ``zigzag_retracement``: direction-normalized movement away from the latest
      pivot, calculated as ``-(close - last) / (last - previous)``. Zero is the
      latest pivot price, one is the preceding pivot price, positive values are
      retracements, and negative values extend the latest swing.

    The output preserves the canonical input index and does not mutate ``data``.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"high", "low", "close"})

    return apply_by_ticker(
        data,
        lambda group: _zigzag_features(
            group,
            deviation=deviation,
            pivot_legs=pivot_legs,
        ),
    )


def zigzag_retracement(
    data: pd.DataFrame,
    *,
    deviation: float = 5.0,
    pivot_legs: int = 10,
) -> pd.Series:
    """Return only the point-in-time direction-normalized Zig Zag retracement."""
    return zigzag_features(
        data,
        deviation=deviation,
        pivot_legs=pivot_legs,
    )["zigzag_retracement"]


def zigzag_swing_return_per_bar(
    data: pd.DataFrame,
    *,
    deviation: float = 5.0,
    pivot_legs: int = 10,
) -> pd.Series:
    """Return only the geometric mean return per bar of the latest Zig Zag swing."""
    return zigzag_features(
        data,
        deviation=deviation,
        pivot_legs=pivot_legs,
    )["zigzag_swing_return_per_bar"]


def _zigzag_features(
    data: pd.DataFrame,
    *,
    deviation: float,
    pivot_legs: int,
) -> pd.DataFrame:
    """Calculate point-in-time Zig Zag features for one ordered instrument."""
    state = _confirmed_zigzag_state(
        data,
        deviation=deviation,
        pivot_legs=pivot_legs,
    )

    last_price = state["_zigzag_last_price"]
    previous_price = state["_zigzag_previous_price"]
    last_position = state["_zigzag_last_position"]
    previous_position = state["_zigzag_previous_position"]

    swing_ratio = safe_divide(last_price, previous_price)
    swing_return = swing_ratio.sub(1.0)
    swing_bars = last_position.sub(previous_position)

    valid_per_bar = swing_ratio.gt(0) & swing_bars.gt(0)
    return_per_bar = (
        swing_ratio.pow(1.0 / swing_bars)
        .sub(1.0)
        .where(valid_per_bar)
        .rename("zigzag_swing_return_per_bar")
    )

    current_position = pd.Series(
        range(len(data)),
        index=data.index,
        dtype="float64",
    )
    bars_since_pivot = current_position.sub(last_position)

    retracement = safe_divide(
        -(data["close"] - last_price),
        last_price - previous_price,
    )

    return pd.DataFrame(
        {
            "zigzag_last_direction": state["_zigzag_last_direction"],
            "zigzag_last_swing_return": swing_return,
            "zigzag_last_swing_bars": swing_bars,
            "zigzag_swing_return_per_bar": return_per_bar,
            "zigzag_bars_since_pivot": bars_since_pivot,
            "zigzag_retracement": retracement,
        },
        index=data.index,
    )[_ZIGZAG_FEATURE_COLUMNS]
