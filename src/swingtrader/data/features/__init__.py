"""Model-oriented feature construction.

Feature builders transform raw market data and reusable indicators from
:mod:`swingtrader.indicators` into model-ready columns. They decide which
indicator inputs to use, how indicators are combined and normalized, how
historical context is represented, and what model-facing columns are named.

Feature builders are organized by family and remain independently usable. The
:func:`swingtrader.data.features.pipeline.add_default_features` orchestrator runs
the standard families in a fixed order.
"""

from swingtrader.data.features.momentum import add_momentum_features
from swingtrader.data.features.returns import add_return_features
from swingtrader.data.features.trend import add_trend_features
from swingtrader.data.features.volatility import add_volatility_features

__all__ = [
    "add_momentum_features",
    "add_return_features",
    "add_trend_features",
    "add_volatility_features",
]
