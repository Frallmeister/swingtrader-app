"""Concrete target-variant sets and their supervised tasks.

Each target set is an independent target variant for a different modeling
question. They are not successive versions and none supersedes the others:

- the forward-return set predicts future return magnitude and a fixed-threshold
  classification derived from it;
- the triple-barrier set describes path-dependent barrier outcomes;
- the cross-sectional-return set describes future performance relative to the
  same-date stock universe.

A set contains more than one family only when the families form one coherent
target definition, such as forward returns and the fixed-threshold class built
directly from them.
"""

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
    FORWARD_RETURN_HORIZONS,
    REQUIRED_PRICE_COLUMNS,
    SIGNIFICANT_RETURN_THRESHOLD,
    TARGET_SIGNIFICANT_UP_5D_COLUMN,
    add_fixed_return_target,
    add_forward_return_targets,
)
from swingtrader.modeling.datasets.triple_barrier import (
    TRIPLE_BARRIER_REQUIRED_PRICE_COLUMNS,
    add_triple_barrier_targets,
    triple_barrier_output_columns,
)

FORWARD_RETURN_TARGET_SET = TargetSetSpec(
    name="forward_return_targets",
    version="1",
    families=(
        TargetFamilySpec(
            name="forward_returns",
            builder=add_forward_return_targets,
            parameters={"horizons": FORWARD_RETURN_HORIZONS},
            required_columns=frozenset(REQUIRED_PRICE_COLUMNS),
            output_columns=FORWARD_RETURN_COLUMNS,
            maximum_horizon_sessions=max(FORWARD_RETURN_HORIZONS),
        ),
        TargetFamilySpec(
            name="significant_up_5d",
            builder=add_fixed_return_target,
            parameters={
                "forward_return_column": "forward_return_5d",
                "output_column": TARGET_SIGNIFICANT_UP_5D_COLUMN,
                "threshold": SIGNIFICANT_RETURN_THRESHOLD,
            },
            required_columns=frozenset({"forward_return_5d"}),
            output_columns=(TARGET_SIGNIFICANT_UP_5D_COLUMN,),
            maximum_horizon_sessions=5,
        ),
    ),
)

FORWARD_RETURN_PRIMARY_TASK = SupervisedTaskSpec(
    name="significant_up_5d_classification",
    target_set_name=FORWARD_RETURN_TARGET_SET.name,
    target_set_version=FORWARD_RETURN_TARGET_SET.version,
    target_column=TARGET_SIGNIFICANT_UP_5D_COLUMN,
    task_type="classification",
    horizon_sessions=5,
)
FORWARD_RETURN_PRIMARY_TASK.validate_target_set(FORWARD_RETURN_TARGET_SET)


TRIPLE_BARRIER_ATR_LENGTH = 14
TRIPLE_BARRIER_STOP_ATR_MULTIPLE = 2.0
TRIPLE_BARRIER_REWARD_RISK_RATIO = 2.0
TRIPLE_BARRIER_HORIZONS = (5, 10, 15)
TRIPLE_BARRIER_INTRABAR_POLICY = "stop_loss_first"
TRIPLE_BARRIER_OUTPUT_COLUMNS = triple_barrier_output_columns(TRIPLE_BARRIER_HORIZONS)

TRIPLE_BARRIER_TARGET_SET = TargetSetSpec(
    name="triple_barrier_targets",
    version="1",
    families=(
        TargetFamilySpec(
            name="triple_barrier",
            builder=add_triple_barrier_targets,
            parameters={
                "atr_length": TRIPLE_BARRIER_ATR_LENGTH,
                "stop_atr_multiple": TRIPLE_BARRIER_STOP_ATR_MULTIPLE,
                "reward_risk_ratio": TRIPLE_BARRIER_REWARD_RISK_RATIO,
                "horizons": TRIPLE_BARRIER_HORIZONS,
                "intrabar_policy": TRIPLE_BARRIER_INTRABAR_POLICY,
            },
            required_columns=frozenset(TRIPLE_BARRIER_REQUIRED_PRICE_COLUMNS),
            output_columns=TRIPLE_BARRIER_OUTPUT_COLUMNS,
            maximum_horizon_sessions=max(TRIPLE_BARRIER_HORIZONS),
        ),
    ),
)

TRIPLE_BARRIER_PRIMARY_TASK = SupervisedTaskSpec(
    name="triple_barrier_5d_classification",
    target_set_name=TRIPLE_BARRIER_TARGET_SET.name,
    target_set_version=TRIPLE_BARRIER_TARGET_SET.version,
    target_column="triple_barrier_label_5d",
    task_type="classification",
    horizon_sessions=5,
    target_end_date_column="target_end_date_5d",
)
TRIPLE_BARRIER_PRIMARY_TASK.validate_target_set(TRIPLE_BARRIER_TARGET_SET)


CROSS_SECTIONAL_HORIZONS = (5, 10, 15)
CROSS_SECTIONAL_RELEVANCE_GRADE_COUNT = 5
CROSS_SECTIONAL_MINIMUM_CROSS_SECTION_SIZE = 2
CROSS_SECTIONAL_OUTPUT_COLUMNS = cross_sectional_return_target_columns(CROSS_SECTIONAL_HORIZONS)

CROSS_SECTIONAL_RETURN_TARGET_SET = TargetSetSpec(
    name="cross_sectional_return_targets",
    version="1",
    families=(
        TargetFamilySpec(
            name="cross_sectional_returns",
            builder=add_cross_sectional_return_targets,
            parameters={
                "horizons": CROSS_SECTIONAL_HORIZONS,
                "relevance_grade_count": CROSS_SECTIONAL_RELEVANCE_GRADE_COUNT,
                "minimum_cross_section_size": CROSS_SECTIONAL_MINIMUM_CROSS_SECTION_SIZE,
            },
            required_columns=frozenset({"adjusted_close"}),
            output_columns=CROSS_SECTIONAL_OUTPUT_COLUMNS,
            maximum_horizon_sessions=max(CROSS_SECTIONAL_HORIZONS),
        ),
    ),
)

CROSS_SECTIONAL_RETURN_PRIMARY_TASK = SupervisedTaskSpec(
    name="forward_return_5d_cross_sectional_percentile_regression",
    target_set_name=CROSS_SECTIONAL_RETURN_TARGET_SET.name,
    target_set_version=CROSS_SECTIONAL_RETURN_TARGET_SET.version,
    target_column="forward_return_5d_cross_sectional_percentile",
    task_type="regression",
    horizon_sessions=5,
)
CROSS_SECTIONAL_RETURN_PRIMARY_TASK.validate_target_set(CROSS_SECTIONAL_RETURN_TARGET_SET)
