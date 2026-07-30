"""Compatibility helpers for executing versioned feature-set contracts."""

import pandas as pd

from swingtrader.data.features.catalog import DEFAULT_FEATURE_SET
from swingtrader.data.features.contracts import FeatureSetSpec


def add_feature_set(
    data: pd.DataFrame,
    *,
    feature_set: FeatureSetSpec = DEFAULT_FEATURE_SET,
) -> pd.DataFrame:
    """Return data with the features enforced by ``feature_set`` appended."""
    return feature_set.apply(data)


def add_default_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return data with the versioned default candidate feature set added."""
    return DEFAULT_FEATURE_SET.apply(data)
