"""Concrete versioned target sets and supervised tasks."""

from swingtrader.modeling.datasets.barriers import (
    BARRIER_REQUIRED_PRICE_COLUMNS,
    add_atr_barrier_targets,
    barrier_output_columns,
)
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


V2_ATR_LENGTH = 14
V2_STOP_ATR_MULTIPLE = 2.0
V2_REWARD_RISK_RATIO = 2.0
V2_BARRIER_HORIZONS = (5, 10, 15)
V2_ENTRY_PRICE_RULE = "next_open"
V2_INTRABAR_POLICY = "exclude_ambiguous"
V2_BARRIER_OUTPUT_COLUMNS = barrier_output_columns(V2_BARRIER_HORIZONS)

V2_TARGET_SET = TargetSetSpec(
    name="ohlcv_price_targets",
    version="2",
    families=(
        *V1_TARGET_SET.families,
        TargetFamilySpec(
            name="atr_barrier_events",
            builder=add_atr_barrier_targets,
            parameters={
                "atr_length": V2_ATR_LENGTH,
                "stop_atr_multiple": V2_STOP_ATR_MULTIPLE,
                "reward_risk_ratio": V2_REWARD_RISK_RATIO,
                "horizons": V2_BARRIER_HORIZONS,
                "entry_price_rule": V2_ENTRY_PRICE_RULE,
                "intrabar_policy": V2_INTRABAR_POLICY,
            },
            required_columns=frozenset(BARRIER_REQUIRED_PRICE_COLUMNS),
            output_columns=V2_BARRIER_OUTPUT_COLUMNS,
            maximum_horizon_sessions=max(V2_BARRIER_HORIZONS),
        ),
    ),
)

V2_PRIMARY_TASK = SupervisedTaskSpec(
    name="tp_before_sl_5d_classification",
    target_set_name=V2_TARGET_SET.name,
    target_set_version=V2_TARGET_SET.version,
    target_column="target_tp_before_sl_5d",
    task_type="classification",
)
V2_PRIMARY_TASK.validate_target_set(V2_TARGET_SET)
