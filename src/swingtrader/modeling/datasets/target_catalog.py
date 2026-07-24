"""Concrete versioned target sets and supervised tasks."""

from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.labels import (
    FORWARD_RETURN_COLUMNS,
    REQUIRED_PRICE_COLUMNS,
    TARGET_SIGNIFICANT_UP_5D_COLUMN,
    V1_FORWARD_RETURN_HORIZONS,
    V1_RETURN_THRESHOLD,
    add_fixed_return_target,
    add_forward_return_targets,
)

V1_TARGET_SET = TargetSetSpec(
    name="ohlcv_price_targets",
    version="1",
    families=(
        TargetFamilySpec(
            name="forward_returns",
            builder=add_forward_return_targets,
            parameters={"horizons": V1_FORWARD_RETURN_HORIZONS},
            required_columns=frozenset(REQUIRED_PRICE_COLUMNS),
            output_columns=FORWARD_RETURN_COLUMNS,
            maximum_horizon_sessions=max(V1_FORWARD_RETURN_HORIZONS),
        ),
        TargetFamilySpec(
            name="significant_up_5d",
            builder=add_fixed_return_target,
            parameters={
                "forward_return_column": "forward_return_5d",
                "output_column": TARGET_SIGNIFICANT_UP_5D_COLUMN,
                "threshold": V1_RETURN_THRESHOLD,
            },
            required_columns=frozenset({"forward_return_5d"}),
            output_columns=(TARGET_SIGNIFICANT_UP_5D_COLUMN,),
            maximum_horizon_sessions=5,
        ),
    ),
)

V1_PRIMARY_TASK = SupervisedTaskSpec(
    name="significant_up_5d_classification",
    target_set_name=V1_TARGET_SET.name,
    target_set_version=V1_TARGET_SET.version,
    target_column=TARGET_SIGNIFICANT_UP_5D_COLUMN,
    task_type="classification",
)
V1_PRIMARY_TASK.validate_target_set(V1_TARGET_SET)
