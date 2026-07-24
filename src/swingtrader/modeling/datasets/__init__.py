"""Dataset construction helpers and versioned target contracts."""

from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.labels import (
    V1_ANNUAL_RETURN_TARGET,
    V1_COMMISSION,
    V1_FORWARD_RETURN_HORIZONS,
    V1_PREDICTION_HORIZON,
    V1_REQUIRED_NET_RETURN,
    V1_RETURN_THRESHOLD,
    V1_TRADING_DAYS_PER_YEAR,
    generate_target_set,
    generate_v1_labels,
)
from swingtrader.modeling.datasets.catalog import V1_PRIMARY_TASK, V1_TARGET_SET

__all__ = [
    "SupervisedTaskSpec",
    "TargetFamilySpec",
    "TargetSetSpec",
    "V1_ANNUAL_RETURN_TARGET",
    "V1_COMMISSION",
    "V1_FORWARD_RETURN_HORIZONS",
    "V1_PREDICTION_HORIZON",
    "V1_PRIMARY_TASK",
    "V1_REQUIRED_NET_RETURN",
    "V1_RETURN_THRESHOLD",
    "V1_TARGET_SET",
    "V1_TRADING_DAYS_PER_YEAR",
    "generate_target_set",
    "generate_v1_labels",
]
