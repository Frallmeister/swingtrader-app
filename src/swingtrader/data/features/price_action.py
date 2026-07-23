"""Price-action features derived from daily OHLC candle geometry.

This module converts reusable candlestick indicators into scale-independent,
point-in-time model features. The feature builder owns the choice to calculate
cross-session quantities on adjustment-consistent OHLC values, while the
indicator layer remains reusable for raw or otherwise transformed prices.
"""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)
from swingtrader.indicators import (
    candle_direction_runs,
    candle_geometry,
    candle_patterns,
    candle_range_context,
    rolling_level_interactions,
)

_GEOMETRY_FEATURE_NAMES = {
    "signed_body_fraction": "candle_signed_body_fraction",
    "upper_wick_fraction": "candle_upper_wick_fraction",
    "lower_wick_fraction": "candle_lower_wick_fraction",
    "close_location": "candle_close_location",
}

_PATTERN_FEATURE_NAMES = {
    "inside_bar": "candle_inside_bar",
    "outside_bar": "candle_outside_bar",
    "engulfing_strength": "candle_engulfing_strength",
    "lower_rejection_strength": "candle_lower_rejection_strength",
    "upper_rejection_strength": "candle_upper_rejection_strength",
    "consecutive_inside_bars": "candle_consecutive_inside_bars",
}

_RUN_FEATURE_NAMES = {
    "direction_run": "candle_direction_run",
    "direction_run_return": "candle_direction_run_return",
    "direction_run_body_atr": "candle_direction_run_body_atr",
}


def add_price_action_features(
    data: pd.DataFrame,
    *,
    atr_length: int = 14,
    range_percentile_length: int = 20,
    breakout_length: int = 20,
) -> pd.DataFrame:
    """Return a copy of data with the default price-action features added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, and
    contain ``open``, ``high``, ``low``, ``close``, and ``adjusted_close``.

    Four continuous candle-geometry features describe the signed real body,
    upper wick, lower wick, and close location as fractions of the current
    high-low range. ``candle_range_atr`` measures current True Range relative to
    the ATR known on the previous row, and ``candle_gap_atr`` measures the signed
    opening gap relative to the same prior ATR. ``range_percentile_{length}``
    ranks the current high-low range against the preceding ``length`` ranges,
    excluding the current row from its reference history. Local pattern outputs
    identify inside and outside bars, count consecutive inside bars, and measure
    engulfing and wick-rejection strength without applying textbook thresholds.
    Directional-run outputs describe the signed same-direction streak, its
    cumulative close-to-close return, and its cumulative signed real-body
    magnitude normalized by prior ATR. Rolling-level outputs measure the close's
    distance from the preceding ``breakout_length``-row high and low, accepted
    breakout penetration, and failed intraday breaks that close back inside the
    prior range. The current
    row is excluded from both rolling levels.

    The price columns are first placed on the ``adjusted_close`` scale by
    multiplying every OHLC value by ``adjusted_close / close``. Same-row geometry
    ratios are unchanged by this transformation, while cross-session gap and ATR
    calculations avoid artificial discontinuities from splits and dividends.
    The standalone indicators remain source-agnostic and can be called directly
    when raw-price outputs are required.

    The feature set favors continuous measurements and direct containment flags
    over thresholded textbook candle labels, making it suitable for model
    training, screening, API responses, and later trade or backtest analysis.
    Warm-up rows remain missing where a calculation needs prior history. The
    candle-geometry features are also missing for zero-range candles. The input
    dataframe is not mutated.
    """
    validate_market_price_index(data)
    validate_required_columns(
        data,
        required_columns={"open", "high", "low", "close", "adjusted_close"},
    )

    range_percentile_name = f"range_percentile_{range_percentile_length}"
    level_feature_names = {
        "close_to_upper_atr": f"candle_close_to_prior_high_atr_{breakout_length}",
        "close_to_lower_atr": f"candle_close_to_prior_low_atr_{breakout_length}",
        "breakout_high_strength": f"candle_breakout_high_strength_{breakout_length}",
        "breakout_low_strength": f"candle_breakout_low_strength_{breakout_length}",
        "failed_break_high_strength": f"candle_failed_breakout_high_strength_{breakout_length}",
        "failed_break_low_strength": f"candle_failed_breakout_low_strength_{breakout_length}",
    }
    new_columns = [
        *_GEOMETRY_FEATURE_NAMES.values(),
        "candle_range_atr",
        "candle_gap_atr",
        range_percentile_name,
        *_PATTERN_FEATURE_NAMES.values(),
        *_RUN_FEATURE_NAMES.values(),
        *level_feature_names.values(),
    ]
    validate_new_columns(data, new_columns=new_columns)

    adjusted_ohlc = _adjusted_ohlc(data)
    level_context = rolling_level_interactions(
        adjusted_ohlc,
        length=breakout_length,
        atr_length=atr_length,
    ).loc[:, list(level_feature_names)]
    level_context = level_context.rename(columns=level_feature_names)
    geometry = candle_geometry(adjusted_ohlc).rename(columns=_GEOMETRY_FEATURE_NAMES)
    range_context = candle_range_context(
        adjusted_ohlc,
        atr_length=atr_length,
        percentile_length=range_percentile_length,
    ).rename(
        columns={
            "range_atr": "candle_range_atr",
            "gap_atr": "candle_gap_atr",
            "range_percentile": range_percentile_name,
        }
    )
    patterns = candle_patterns(adjusted_ohlc, atr_length=atr_length).rename(
        columns=_PATTERN_FEATURE_NAMES
    )
    direction_runs = candle_direction_runs(
        adjusted_ohlc,
        atr_length=atr_length,
    ).rename(columns=_RUN_FEATURE_NAMES)

    result = data.copy()
    return (
        result.join(geometry)
        .join(range_context)
        .join(patterns)
        .join(direction_runs)
        .join(level_context)
    )


def _adjusted_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    adjustment_factor = safe_divide(data.loc[:, "adjusted_close"], data.loc[:, "close"])
    return data.loc[:, ["open", "high", "low", "close"]].mul(adjustment_factor, axis=0)
