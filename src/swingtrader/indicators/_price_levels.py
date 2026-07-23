"""Private calculations shared by price-level interaction consumers."""

import numpy as np
import pandas as pd

from swingtrader.core.numerical import safe_divide


def _price_level_interactions(
    data: pd.DataFrame,
    *,
    upper_level: pd.Series,
    lower_level: pd.Series,
    prior_atr: pd.Series,
) -> pd.DataFrame:
    """Measure close acceptance and intraday rejection around two price levels."""
    high = data.loc[:, "high"]
    low = data.loc[:, "low"]
    close = data.loc[:, "close"]

    close_to_upper_atr = safe_divide(close - upper_level, prior_atr)
    close_to_lower_atr = safe_divide(close - lower_level, prior_atr)

    breakout_high_strength = close_to_upper_atr.clip(lower=0.0)
    breakout_low_strength = close_to_lower_atr.mul(-1.0).clip(lower=0.0)

    high_excursion_atr = safe_divide(high - upper_level, prior_atr)
    low_excursion_atr = safe_divide(lower_level - low, prior_atr)

    failed_break_high = high.gt(upper_level) & close.le(upper_level)
    failed_break_low = low.lt(lower_level) & close.ge(lower_level)

    valid_high_failure = high_excursion_atr.notna() & np.isfinite(close)
    valid_low_failure = low_excursion_atr.notna() & np.isfinite(close)

    failed_break_high_strength = high_excursion_atr.where(
        failed_break_high,
        0.0,
    ).where(valid_high_failure)
    failed_break_low_strength = low_excursion_atr.where(
        failed_break_low,
        0.0,
    ).where(valid_low_failure)

    return pd.DataFrame(
        {
            "close_to_upper_atr": close_to_upper_atr,
            "close_to_lower_atr": close_to_lower_atr,
            "breakout_high_strength": breakout_high_strength,
            "breakout_low_strength": breakout_low_strength,
            "failed_break_high_strength": failed_break_high_strength,
            "failed_break_low_strength": failed_break_low_strength,
        },
        index=data.index,
    )
