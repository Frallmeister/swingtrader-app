from datetime import date

import pandas as pd
import pytest

from swingtrader.data.features.contracts import FeatureBlockSpec, FeatureSetSpec
from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.specifications import UniverseSpec as DatasetUniverseSpec
from swingtrader.modeling.experiments import UniverseSpec
from swingtrader.modeling.experiments.contracts import (
    ExperimentSpec,
    ModelSpec,
    TemporalSplitSpec,
)


def add_feature(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(feature=data["value"])


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    return data.assign(target=True)


def test_experiment_exposes_lower_level_dataset_spec() -> None:
    feature_set = FeatureSetSpec(
        name="features",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="feature",
                builder=add_feature,
                required_columns=frozenset({"value"}),
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
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target",
        task_type="classification",
        horizon_sessions=1,
    )
    universe = UniverseSpec(
        name="universe",
        version="1",
        provider="test",
        tickers=("AAA",),
    )
    experiment = ExperimentSpec(
        name="experiment",
        version="1",
        feature_set=feature_set,
        target_set=target_set,
        task=task,
        universe=universe,
        data_cutoff=date(2026, 1, 31),
        split=TemporalSplitSpec(
            name="split",
            version="1",
            train_start=date(2025, 1, 1),
            train_end=date(2025, 6, 30),
            validation_start=date(2025, 7, 1),
            validation_end=date(2025, 9, 30),
            test_start=date(2025, 10, 1),
            test_end=date(2026, 1, 31),
        ),
        model=ModelSpec(
            name="model",
            version="1",
            model_type="example.Model",
        ),
        random_seeds={"model": 17},
    )

    assert UniverseSpec is DatasetUniverseSpec
    assert experiment.dataset_spec.feature_set is feature_set
    assert experiment.dataset_spec.target_set is target_set
    assert experiment.dataset_spec.universe is universe
    manifest = experiment.dataset_spec.to_manifest()
    forbidden_keys = {
        "split",
        "temporal_split",
        "model",
        "model_spec",
        "random_seed",
        "random_seeds",
        "mlflow",
    }
    assert forbidden_keys.isdisjoint(manifest)

    horizonless_task = SupervisedTaskSpec(
        name="horizonless_task",
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target",
        task_type="classification",
    )
    with pytest.raises(ValueError, match="horizon_sessions"):
        ExperimentSpec(
            name="invalid_experiment",
            version="1",
            feature_set=feature_set,
            target_set=target_set,
            task=horizonless_task,
            universe=universe,
            data_cutoff=experiment.data_cutoff,
            split=experiment.split,
            model=experiment.model,
            random_seeds=experiment.random_seeds,
        )
