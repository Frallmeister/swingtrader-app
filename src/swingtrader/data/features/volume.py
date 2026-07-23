"""Volume-based model features.

This module adds point-in-time-safe features derived from traded volume and
price to canonical market-price dataframes. Calculations are isolated within
each provider/ticker group, preserve the original index and row order, and do
not mutate the supplied dataframe.
"""

import pandas as pd

from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)
from swingtrader.indicators import turnover_zscore


def add_volume_features(
    data: pd.DataFrame,
    *,
    turnover_zscore_length: int = 252,
    turnover_zscore_log: bool = True,
) -> pd.DataFrame:
    """Add volume-based model features to canonical market-price data.

    The function adds ``turnover_zscore``, which measures how unusual the current
    turnover is relative to the preceding ``turnover_zscore_length - 1``
    observations for the same provider and ticker. Turnover is calculated as
    ``adjusted_close * volume``.

    By default, turnover is transformed with ``log1p`` before normalization. This
    reduces its right skew and limits the influence of extreme turnover values.
    The current observation is excluded from its own reference statistics, making
    the feature point-in-time safe.

    ``data`` must use the canonical ``provider``, ``ticker``, and
    ``trading_date`` index levels and contain ``adjusted_close`` and ``volume``
    columns. Calculations are isolated within each provider/ticker group. The
    input dataframe is copied before the feature is added.

    Parameters
    ----------
    data
        Canonical market-price dataframe containing ``adjusted_close`` and
        ``volume`` columns.
    turnover_zscore_length
        Total observation span used by the turnover z-score, including the
        current row. The historical reference window contains one fewer row.
    turnover_zscore_log
        Whether to apply ``log1p`` to turnover before normalization.

    Returns
    -------
    pandas.DataFrame
        A copy of ``data`` with the ``turnover_zscore`` column added.

    Raises
    ------
    ValueError
        If the canonical market-price index is invalid, the z-score length is
        invalid or less than 2, or ``turnover_zscore_log`` is not a boolean.
    KeyError
        If a required input column is missing.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"adjusted_close", "volume"})
    validate_new_columns(data, new_columns={"turnover_zscore"})
    data = data.copy()
    data["turnover_zscore"] = turnover_zscore(
        data,
        length=turnover_zscore_length,
        log=turnover_zscore_log,
    )
    return data
