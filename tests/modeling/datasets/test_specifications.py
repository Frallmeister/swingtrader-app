from datetime import date

import pandas as pd
import pytest

from swingtrader.data.features.contracts import FeatureBlockSpec, FeatureSetSpec
from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.specifications import TemporalDatasetSpec, UniverseSpec


def add_feature(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(feature=data["source"])


def add_intermediate_feature(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(intermediate=data["source"])


def add_dependent_feature(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(feature=data["intermediate"])


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(target=True)


def add_dependent_target(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(derived_target=data["target"])


def test_source_columns_exclude_outputs_produced_earlier() -> None:
    feature_set = FeatureSetSpec(
        name="features",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="intermediate",
                builder=add_intermediate_feature,
                required_columns=frozenset({"source"}),
                output_columns=("intermediate",),
            ),
            FeatureBlockSpec(
                name="dependent",
                builder=add_dependent_feature,
                required_columns=frozenset({"intermediate"}),
                output_columns=("feature",),
            ),
        ),
    )
    target_set = TargetSetSpec(
        name="targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="target",
                builder=add_target,
                required_columns=frozenset({"source"}),
                output_columns=("target",),
                maximum_horizon_sessions=1,
            ),
            TargetFamilySpec(
                name="derived",
                builder=add_dependent_target,
                required_columns=frozenset({"target"}),
                output_columns=("derived_target",),
                maximum_horizon_sessions=1,
            ),
        ),
    )

    assert feature_set.source_columns == ("source",)
    assert target_set.source_columns == ("source",)


def test_task_validates_horizon_and_explicit_end_date_column() -> None:
    target_set = TargetSetSpec(
        name="targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="target",
                builder=add_target,
                output_columns=("target",),
                maximum_horizon_sessions=2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unknown target end-date column"):
        SupervisedTaskSpec(
            name="task",
            target_set_name="targets",
            target_set_version="1",
            target_column="target",
            task_type="classification",
            horizon_sessions=2,
            target_end_date_column="missing",
        ).validate_target_set(target_set)


def test_task_rejects_target_column_as_end_date_column() -> None:
    with pytest.raises(ValueError, match="must differ"):
        SupervisedTaskSpec(
            name="task",
            target_set_name="targets",
            target_set_version="1",
            target_column="target",
            task_type="classification",
            horizon_sessions=1,
            target_end_date_column="target",
        )


def test_temporal_dataset_spec_requires_task_horizon() -> None:
    feature_set = FeatureSetSpec(
        name="features",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="feature",
                builder=add_feature,
                required_columns=frozenset({"source"}),
                output_columns=("feature",),
            ),
        ),
    )
    target_set = TargetSetSpec(
        name="targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="target",
                builder=add_target,
                output_columns=("target",),
                maximum_horizon_sessions=1,
            ),
        ),
    )
    task = SupervisedTaskSpec(
        name="task",
        target_set_name="targets",
        target_set_version="1",
        target_column="target",
        task_type="classification",
    )

    with pytest.raises(ValueError, match="horizon_sessions"):
        TemporalDatasetSpec(
            feature_set=feature_set,
            target_set=target_set,
            task=task,
            universe=UniverseSpec(name="universe", version="1", provider="test", tickers=("AAA",)),
            data_start=date(2020, 1, 1),
            data_end=date(2026, 1, 1),
        )
