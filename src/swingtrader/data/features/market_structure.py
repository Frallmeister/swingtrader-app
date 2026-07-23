"""Point-in-time market-structure feature transformations.

This module converts confirmed Zig Zag state and swing-level interactions into
normalized, row-aligned features. Unlike the retrospective
:func:`swingtrader.indicators.zigzag` output,
these features update only when each pivot has enough right-side observations to
be confirmed. Intermediate endpoints remain visible until a later confirmed,
more-extreme endpoint replaces them.
"""

import numpy as np
import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import (
    apply_by_ticker,
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)
from swingtrader.indicators._price_levels import _price_level_interactions
from swingtrader.indicators._validation import validate_length
from swingtrader.indicators.market_structure import _confirmed_zigzag_state
from swingtrader.indicators.volatility import _atr

_MARKET_STRUCTURE_FEATURE_COLUMNS = (
    "zigzag_last_direction",
    "zigzag_last_swing_return",
    "zigzag_last_swing_bars",
    "zigzag_swing_return_per_bar",
    "zigzag_bars_since_pivot",
    "zigzag_retracement",
    "market_structure_high_change",
    "market_structure_low_change",
    "market_structure_high_rate",
    "market_structure_low_rate",
    "market_structure_high_consistency",
    "market_structure_low_consistency",
    "market_structure_leg_balance",
    "market_structure_efficiency",
    "market_structure_close_to_prior_high_atr",
    "market_structure_close_to_prior_low_atr",
    "market_structure_breakout_high_strength",
    "market_structure_breakout_low_strength",
    "market_structure_failed_breakout_high_strength",
    "market_structure_failed_breakout_low_strength",
)


def add_market_structure_features(
    data: pd.DataFrame,
    *,
    zigzag_deviation: float = 5.0,
    zigzag_pivot_legs: int = 10,
    zigzag_consistency_pivots: int = 4,
    zigzag_dynamics_legs: int = 6,
    zigzag_atr_length: int = 14,
) -> pd.DataFrame:
    """Return a copy of data with the default market-structure features added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date`` and contain ``high``, ``low``,
    and ``close`` columns. The appended features are point-in-time: a Zig Zag
    pivot and its associated support or resistance level affect the output only
    on and after the pivot confirmation row.

    Use :func:`zigzag_features` directly when only the feature block is needed,
    for example for a frontend endpoint that should not calculate every feature
    family. Existing columns with the generated feature names are rejected rather
    than silently overwritten.
    """
    validate_new_columns(data, new_columns=_MARKET_STRUCTURE_FEATURE_COLUMNS)
    result = data.copy()
    feature_block = zigzag_features(
        data,
        deviation=zigzag_deviation,
        pivot_legs=zigzag_pivot_legs,
        consistency_pivots=zigzag_consistency_pivots,
        dynamics_legs=zigzag_dynamics_legs,
        atr_length=zigzag_atr_length,
    )
    result[feature_block.columns] = feature_block
    return result


def zigzag_features(
    data: pd.DataFrame,
    *,
    deviation: float = 5.0,
    pivot_legs: int = 10,
    consistency_pivots: int = 4,
    dynamics_legs: int = 6,
    atr_length: int = 14,
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
      retracements, and negative values extend the latest swing;
    - ``market_structure_high_change`` and ``market_structure_low_change``:
      logarithmic price changes between the latest two confirmed swing highs and
      latest two confirmed swing lows, respectively;
    - ``market_structure_high_rate`` and ``market_structure_low_rate``: the
      corresponding logarithmic changes divided by the number of input rows
      between the two historical pivot positions;
    - ``market_structure_high_consistency`` and
      ``market_structure_low_consistency``: Kendall's tau-b between chronological
      order and the prices of the latest ``consistency_pivots`` confirmed swing
      highs or lows. Values range from ``-1`` for consistently falling pivots to
      ``1`` for consistently rising pivots;
    - ``market_structure_leg_balance``: median upward completed-leg magnitude
      minus median downward magnitude, divided by their sum. Values range from
      ``-1`` when downward legs dominate to ``1`` when upward legs dominate;
    - ``market_structure_efficiency``: signed net log displacement divided by
      total absolute log path length over the latest ``dynamics_legs`` completed
      legs. Values near zero indicate substantial movement with little net
      progress.
    - ``market_structure_close_to_prior_high_atr`` and
      ``market_structure_close_to_prior_low_atr``: close distance from the latest
      confirmed swing high and low, normalized by prior ATR;
    - ``market_structure_breakout_high_strength`` and
      ``market_structure_breakout_low_strength``: positive close penetration beyond
      the latest confirmed swing level, normalized by prior ATR;
    - ``market_structure_failed_breakout_high_strength`` and
      ``market_structure_failed_breakout_low_strength``: positive intraday excursion
      beyond a confirmed level when the close finishes back on the other side.

    The structural changes and rates remain missing until two confirmed pivots of
    the corresponding direction are available. Consistency remains missing until
    ``consistency_pivots`` same-direction pivots are available, and is also missing
    when all selected prices are equal. Leg balance and efficiency remain
    missing until ``dynamics_legs`` completed legs are available. Swing-level
    interactions remain missing until the corresponding confirmed level and prior
    ATR are available; evaluable rows without a break receive zero strength. The
    dynamics window must be even so it contains equal numbers of upward and
    downward legs. The output preserves the canonical input index and does not
    mutate ``data``.

    Notes
    -----
    All returned columns are point-in-time safe for row-aligned modeling. Pivot
    information first appears on its confirmation row; future rows never revise
    previously emitted feature values. Leg dynamics use only adjacent confirmed
    endpoints and exclude movement from the latest endpoint toward the current
    close or an interpolated active leg.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"high", "low", "close"})
    validate_length(atr_length)

    return apply_by_ticker(
        data,
        lambda group: _zigzag_features(
            group,
            deviation=deviation,
            pivot_legs=pivot_legs,
            consistency_pivots=consistency_pivots,
            dynamics_legs=dynamics_legs,
            atr_length=atr_length,
        ),
    )


def _zigzag_features(
    data: pd.DataFrame,
    *,
    deviation: float,
    pivot_legs: int,
    consistency_pivots: int,
    dynamics_legs: int,
    atr_length: int,
) -> pd.DataFrame:
    """Calculate point-in-time Zig Zag features for one ordered instrument."""
    state = _confirmed_zigzag_state(
        data,
        deviation=deviation,
        pivot_legs=pivot_legs,
        consistency_pivots=consistency_pivots,
        dynamics_legs=dynamics_legs,
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

    high_change, high_rate = _structural_change_and_rate(
        state["_zigzag_last_high_price"],
        state["_zigzag_previous_high_price"],
        state["_zigzag_last_high_position"],
        state["_zigzag_previous_high_position"],
        change_name="market_structure_high_change",
        rate_name="market_structure_high_rate",
    )
    low_change, low_rate = _structural_change_and_rate(
        state["_zigzag_last_low_price"],
        state["_zigzag_previous_low_price"],
        state["_zigzag_last_low_position"],
        state["_zigzag_previous_low_position"],
        change_name="market_structure_low_change",
        rate_name="market_structure_low_rate",
    )

    level_interactions = _price_level_interactions(
        data,
        upper_level=state["_zigzag_last_high_price"],
        lower_level=state["_zigzag_last_low_price"],
        prior_atr=_atr(data, length=atr_length).shift(1),
    )

    return pd.DataFrame(
        {
            "zigzag_last_direction": state["_zigzag_last_direction"],
            "zigzag_last_swing_return": swing_return,
            "zigzag_last_swing_bars": swing_bars,
            "zigzag_swing_return_per_bar": return_per_bar,
            "zigzag_bars_since_pivot": bars_since_pivot,
            "zigzag_retracement": retracement,
            "market_structure_high_change": high_change,
            "market_structure_low_change": low_change,
            "market_structure_high_rate": high_rate,
            "market_structure_low_rate": low_rate,
            "market_structure_high_consistency": state["_zigzag_high_consistency"],
            "market_structure_low_consistency": state["_zigzag_low_consistency"],
            "market_structure_leg_balance": state["_zigzag_leg_balance"],
            "market_structure_efficiency": state["_zigzag_efficiency"],
            "market_structure_close_to_prior_high_atr": level_interactions["close_to_upper_atr"],
            "market_structure_close_to_prior_low_atr": level_interactions["close_to_lower_atr"],
            "market_structure_breakout_high_strength": level_interactions["breakout_high_strength"],
            "market_structure_breakout_low_strength": level_interactions["breakout_low_strength"],
            "market_structure_failed_breakout_high_strength": level_interactions[
                "failed_break_high_strength"
            ],
            "market_structure_failed_breakout_low_strength": level_interactions[
                "failed_break_low_strength"
            ],
        },
        index=data.index,
    )


def _structural_change_and_rate(
    last_price: pd.Series,
    previous_price: pd.Series,
    last_position: pd.Series,
    previous_position: pd.Series,
    *,
    change_name: str,
    rate_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Calculate log displacement and average log displacement per row."""
    price_ratio = safe_divide(last_price, previous_price)
    change = np.log(price_ratio.where(price_ratio.gt(0))).rename(change_name)
    bars = last_position.sub(previous_position)
    rate = safe_divide(change, bars.where(bars.gt(0))).rename(rate_name)
    return change, rate
