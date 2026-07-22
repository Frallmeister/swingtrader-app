"""Default feature pipeline orchestration.

This module exposes a single explicit entry point that runs the standard feature
families in a fixed order. It contains no indicator or feature formulas, no
dynamic discovery, and no dependency resolution; it simply calls the existing
family builders in sequence. Each family builder remains independently usable.
"""

import pandas as pd

from swingtrader.data.features.market_structure import add_market_structure_features
from swingtrader.data.features.momentum import add_momentum_features
from swingtrader.data.features.returns import add_return_features
from swingtrader.data.features.trend import add_trend_features
from swingtrader.data.features.volatility import add_volatility_features
from swingtrader.data.features.volume import add_volume_features


def add_default_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of data with the default feature set from every family added.

    The families run in an explicit, stable order: returns, trend, momentum,
    volatility, volume then market structure. The input must satisfy the canonical
    market-price contract and provide every column the individual families
    require, including ``high``, ``low``, ``close``, ``adjusted_close``, and
    ``volume``. The input index and
    row order are preserved and the input dataframe is not mutated. The result is
    equivalent to applying the family builders manually in the same order.
    """
    data = add_return_features(data)
    data = add_trend_features(data)
    data = add_momentum_features(data)
    data = add_volatility_features(data)
    data = add_volume_features(data)
    data = add_market_structure_features(data)
    return data
