"""Feature-set pipeline orchestration.

This module executes explicit, versioned feature-set specifications. It contains
no indicator or feature formulas, dynamic discovery, or dependency resolution.
Each family builder remains independently usable.
"""

import pandas as pd

from swingtrader.data.features.catalog import DEFAULT_FEATURE_SET
from swingtrader.data.features.feature_sets import FeatureSetSpec


def add_feature_set(
    data: pd.DataFrame,
    *,
    feature_set: FeatureSetSpec = DEFAULT_FEATURE_SET,
) -> pd.DataFrame:
    """Return a copy of data with the declared feature blocks added.

    Blocks run in their declared order with the parameters recorded by the
    feature-set specification. Each block retains its own validation and
    point-in-time calculation contract. The input index and row order are
    preserved and the input dataframe is not mutated.
    """
    result = data
    for block in feature_set.blocks:
        result = block.apply(result)
    return result


def add_default_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return data with the versioned default OHLCV candidate set added.

    This compatibility wrapper delegates to :func:`add_feature_set` using
    :data:`swingtrader.data.features.catalog.DEFAULT_FEATURE_SET`.
    """
    return add_feature_set(data)
