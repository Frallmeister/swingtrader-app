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
from swingtrader.indicators import candle_geometry, candle_range_context

_GEOMETRY_FEATURE_NAMES = {
    "signed_body_fraction": "candle_signed_body_fraction",
    "upper_wick_fraction": "candle_upper_wick_fraction",
    "lower_wick_fraction": "candle_lower_wick_fraction",
    "close_location": "candle_close_location",
}


def add_price_action_features(
    data: pd.DataFrame,
    *,
    atr_length: int = 14,
    range_percentile_length: int = 20,
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
    excluding the current row from its reference history.

    The price columns are first placed on the ``adjusted_close`` scale by
    multiplying every OHLC value by ``adjusted_close / close``. Same-row geometry
    ratios are unchanged by this transformation, while cross-session gap and ATR
    calculations avoid artificial discontinuities from splits and dividends.
    The standalone indicators remain source-agnostic and can be called directly
    when raw-price outputs are required.

    The generated features are continuous rather than thresholded candlestick
    labels, making them suitable for model training, screening, API responses,
    and later trade or backtest analysis. Warm-up rows and zero-range candles
    remain missing. The input dataframe is not mutated.
    """
    validate_market_price_index(data)
    validate_required_columns(
        data,
        required_columns={"open", "high", "low", "close", "adjusted_close"},
    )

    range_percentile_name = f"range_percentile_{range_percentile_length}"
    new_columns = [
        *_GEOMETRY_FEATURE_NAMES.values(),
        "candle_range_atr",
        "candle_gap_atr",
        range_percentile_name,
    ]
    validate_new_columns(data, new_columns=new_columns)

    adjusted_ohlc = _adjusted_ohlc(data)
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

    result = data.copy()
    return result.join(geometry).join(range_context)


def _adjusted_ohlc(data: pd.DataFrame) -> pd.DataFrame:
    adjustment_factor = safe_divide(data.loc[:, "adjusted_close"], data.loc[:, "close"])
    return data.loc[:, ["open", "high", "low", "close"]].mul(adjustment_factor, axis=0)
