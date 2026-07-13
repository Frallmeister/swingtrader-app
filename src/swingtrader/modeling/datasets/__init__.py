"""Dataset construction helpers for modeling workflows."""

from swingtrader.modeling.datasets.labels import (
    V1_ANNUAL_RETURN_TARGET,
    V1_COMMISSION,
    V1_FORWARD_RETURN_HORIZONS,
    V1_PREDICTION_HORIZON,
    V1_REQUIRED_NET_RETURN,
    V1_RETURN_THRESHOLD,
    V1_TRADING_DAYS_PER_YEAR,
    generate_v1_labels,
)

__all__ = [
    "V1_ANNUAL_RETURN_TARGET",
    "V1_COMMISSION",
    "V1_FORWARD_RETURN_HORIZONS",
    "V1_PREDICTION_HORIZON",
    "V1_REQUIRED_NET_RETURN",
    "V1_RETURN_THRESHOLD",
    "V1_TRADING_DAYS_PER_YEAR",
    "generate_v1_labels",
]
