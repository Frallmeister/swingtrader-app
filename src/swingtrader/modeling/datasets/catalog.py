"""Concrete versioned target sets and supervised tasks."""

from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.cross_sectional import (
    add_cross_sectional_return_targets,
    cross_sectional_return_target_columns,
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
from swingtrader.modeling.datasets.triple_barrier import (
    TRIPLE_BARRIER_REQUIRED_PRICE_COLUMNS,
    add_triple_barrier_targets,
    triple_barrier_output_columns,
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
    horizon_sessions=5,
)
V1_PRIMARY_TASK.validate_target_set(V1_TARGET_SET)


V3_ATR_LENGTH = 14
V3_STOP_ATR_MULTIPLE = 2.0
V3_REWARD_RISK_RATIO = 2.0
V3_TRIPLE_BARRIER_HORIZONS = (5, 10, 15)
V3_INTRABAR_POLICY = "stop_loss_first"
V3_TRIPLE_BARRIER_OUTPUT_COLUMNS = triple_barrier_output_columns(V3_TRIPLE_BARRIER_HORIZONS)

V3_TARGET_SET = TargetSetSpec(
    name="ohlcv_price_targets",
    version="3",
    families=(
        *V1_TARGET_SET.families,
        TargetFamilySpec(
            name="triple_barrier",
            builder=add_triple_barrier_targets,
            parameters={
                "atr_length": V3_ATR_LENGTH,
                "stop_atr_multiple": V3_STOP_ATR_MULTIPLE,
                "reward_risk_ratio": V3_REWARD_RISK_RATIO,
                "horizons": V3_TRIPLE_BARRIER_HORIZONS,
                "intrabar_policy": V3_INTRABAR_POLICY,
            },
            required_columns=frozenset(TRIPLE_BARRIER_REQUIRED_PRICE_COLUMNS),
            output_columns=V3_TRIPLE_BARRIER_OUTPUT_COLUMNS,
            maximum_horizon_sessions=max(V3_TRIPLE_BARRIER_HORIZONS),
        ),
    ),
)

V3_PRIMARY_TASK = SupervisedTaskSpec(
    name="triple_barrier_5d_classification",
    target_set_name=V3_TARGET_SET.name,
    target_set_version=V3_TARGET_SET.version,
    target_column="triple_barrier_label_5d",
    task_type="classification",
    horizon_sessions=5,
    target_end_date_column="target_end_date_5d",
)
V3_PRIMARY_TASK.validate_target_set(V3_TARGET_SET)


V4_CROSS_SECTIONAL_HORIZONS = V1_FORWARD_RETURN_HORIZONS
V4_RELEVANCE_GRADE_COUNT = 5
V4_CROSS_SECTIONAL_OUTPUT_COLUMNS = cross_sectional_return_target_columns(
    V4_CROSS_SECTIONAL_HORIZONS
)
V4_TARGET_SET = TargetSetSpec(
    name="ohlcv_price_targets",
    version="4",
    families=(
        *V3_TARGET_SET.families,
        TargetFamilySpec(
            name="cross_sectional_returns",
            builder=add_cross_sectional_return_targets,
            parameters={
                "horizons": V4_CROSS_SECTIONAL_HORIZONS,
                "relevance_grade_count": V4_RELEVANCE_GRADE_COUNT,
            },
            required_columns=frozenset(FORWARD_RETURN_COLUMNS),
            output_columns=V4_CROSS_SECTIONAL_OUTPUT_COLUMNS,
            maximum_horizon_sessions=max(V4_CROSS_SECTIONAL_HORIZONS),
        ),
    ),
)
V4_PRIMARY_TASK = SupervisedTaskSpec(
    name="forward_return_5d_cross_sectional_percentile_regression",
    target_set_name=V4_TARGET_SET.name,
    target_set_version=V4_TARGET_SET.version,
    target_column="forward_return_5d_cross_sectional_percentile",
    task_type="regression",
    horizon_sessions=5,
)
V4_PRIMARY_TASK.validate_target_set(V4_TARGET_SET)
