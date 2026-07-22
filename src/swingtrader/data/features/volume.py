"""
MODULE DOCSTRING
"""

import pandas as pd

from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.indicators import turnover_zscore


def add_volume_features(
    data,
    *,
    turnover_zscore_length: int = 252,
    turnover_zscore_log: bool = True,
) -> pd.DataFrame:
    """ADD DOCSTRING HERE."""
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"adjusted_close", "volume"})
    data = data.copy()
    data["turnover_zscore"] = turnover_zscore(
        data,
        length=turnover_zscore_length,
        log=turnover_zscore_log,
    )
    return data
