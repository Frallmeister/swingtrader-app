"""Model-oriented feature construction.

Feature builders transform raw market data and reusable indicators from
:mod:`swingtrader.indicators` into model-ready columns. They decide which
indicator inputs to use, how indicators are combined and normalized, how
historical context is represented, and what model-facing columns are named.

Feature builders remain independently usable. Versioned feature-set contracts
record which builders, parameters, and output columns define reproducible model
inputs.
"""

from swingtrader.data.features.catalog import DEFAULT_FEATURE_SET
from swingtrader.data.features.feature_sets import (
    FeatureBlockSpec,
    FeatureSetSpec,
    HistoryRequirement,
)
from swingtrader.data.features.market_structure import (
    add_market_structure_features,
    zigzag_features,
)
from swingtrader.data.features.momentum import add_momentum_features
from swingtrader.data.features.pipeline import add_default_features, add_feature_set
from swingtrader.data.features.price_action import add_price_action_features
from swingtrader.data.features.returns import add_return_features
from swingtrader.data.features.trend import add_trend_features
from swingtrader.data.features.volatility import add_volatility_features
from swingtrader.data.features.volume import add_volume_features

__all__ = [
    "DEFAULT_FEATURE_SET",
    "FeatureBlockSpec",
    "FeatureSetSpec",
    "HistoryRequirement",
    "add_default_features",
    "add_feature_set",
    "add_market_structure_features",
    "add_momentum_features",
    "add_price_action_features",
    "add_return_features",
    "add_trend_features",
    "add_volatility_features",
    "add_volume_features",
    "zigzag_features",
]
