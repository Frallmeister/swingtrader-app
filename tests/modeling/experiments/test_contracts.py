from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from swingtrader.data.features.catalog import DEFAULT_FEATURE_SET
from swingtrader.modeling.datasets.catalog import V1_PRIMARY_TASK, V1_TARGET_SET
from swingtrader.modeling.experiments import (
    ExperimentSpec,
    ModelSpec,
    TemporalSplitSpec,
    UniverseSpec,
)


def _experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        name="baseline",
        version="1",
        feature_set=DEFAULT_FEATURE_SET,
        target_set=V1_TARGET_SET,
        task=V1_PRIMARY_TASK,
        universe=UniverseSpec(
            name="se_large_mid_cap",
            version="2026-07-24",
            provider="yfinance",
            tickers=("VOLV-B.ST", "ABB.ST"),
        ),
        data_cutoff=date(2025, 12, 31),
        split=TemporalSplitSpec(
            name="holdout",
            version="1",
            train_start=date(2010, 1, 1),
            train_end=date(2021, 12, 31),
            validation_start=date(2022, 1, 1),
            validation_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1),
            test_end=date(2025, 12, 31),
        ),
        model=ModelSpec(
            name="logistic_regression",
            version="1",
            model_type="sklearn.linear_model.LogisticRegression",
            hyperparameters={
                "C": 1.0,
                "class_weight": None,
                "penalty": "l2",
                "solver_options": {"tolerance": 1e-6},
            },
        ),
        random_seeds={"model": 17, "sampling": 23},
    )


def test_experiment_manifest_is_deterministic_and_json_serializable() -> None:
    first = _experiment_spec()
    second = _experiment_spec()

    assert first.to_manifest() == second.to_manifest()
    assert first.digest == second.digest
    assert json.loads(first.to_json()) == first.to_manifest()
    assert first.universe.tickers == ("ABB.ST", "VOLV-B.ST")


def test_meaningful_configuration_change_changes_experiment_digest() -> None:
    original = _experiment_spec()
    changed = ExperimentSpec(
        name=original.name,
        version=original.version,
        feature_set=original.feature_set,
        target_set=original.target_set,
        task=original.task,
        universe=original.universe,
        data_cutoff=original.data_cutoff,
        split=original.split,
        model=ModelSpec(
            name=original.model.name,
            version=original.model.version,
            model_type=original.model.model_type,
            hyperparameters={
                "C": 0.5,
                "class_weight": None,
                "penalty": "l2",
                "solver_options": {"tolerance": 1e-6},
            },
        ),
        random_seeds=original.random_seeds,
    )

    assert changed.digest != original.digest


def test_specs_freeze_caller_owned_collections() -> None:
    tickers = ["VOLV-B.ST", "ABB.ST"]
    hyperparameters = {"layers": [64, 32]}
    seeds = {"model": 17}

    universe = UniverseSpec(
        name="universe",
        version="1",
        provider="yfinance",
        tickers=tickers,  # type: ignore[arg-type]
    )
    model = ModelSpec(
        name="model",
        version="1",
        model_type="example.Model",
        hyperparameters=hyperparameters,
    )
    experiment = _experiment_spec()
    experiment = ExperimentSpec(
        name=experiment.name,
        version=experiment.version,
        feature_set=experiment.feature_set,
        target_set=experiment.target_set,
        task=experiment.task,
        universe=universe,
        data_cutoff=experiment.data_cutoff,
        split=experiment.split,
        model=model,
        random_seeds=seeds,
    )

    tickers.append("SAAB-B.ST")
    hyperparameters["layers"].append(16)  # type: ignore[union-attr]
    seeds["model"] = 99

    assert universe.tickers == ("ABB.ST", "VOLV-B.ST")
    assert model.to_manifest()["hyperparameters"] == {"layers": [64, 32]}
    assert experiment.random_seeds == {"model": 17}
    with pytest.raises(TypeError):
        model.hyperparameters["layers"] = (1,)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        universe.name = "changed"  # type: ignore[misc]


def test_temporal_split_rejects_overlapping_ranges() -> None:
    with pytest.raises(ValueError, match="Training and validation"):
        TemporalSplitSpec(
            name="overlap",
            version="1",
            train_start=date(2020, 1, 1),
            train_end=date(2022, 1, 1),
            validation_start=date(2022, 1, 1),
            validation_end=date(2023, 1, 1),
            test_start=date(2024, 1, 1),
            test_end=date(2025, 1, 1),
        )


def test_experiment_rejects_task_for_another_target_set() -> None:
    experiment = _experiment_spec()
    wrong_task = type(V1_PRIMARY_TASK)(
        name="wrong",
        target_set_name="other",
        target_set_version="1",
        target_column=experiment.task.target_column,
        task_type="classification",
    )

    with pytest.raises(ValueError, match="different target set"):
        ExperimentSpec(
            name=experiment.name,
            version=experiment.version,
            feature_set=experiment.feature_set,
            target_set=experiment.target_set,
            task=wrong_task,
            universe=experiment.universe,
            data_cutoff=experiment.data_cutoff,
            split=experiment.split,
            model=experiment.model,
            random_seeds=experiment.random_seeds,
        )


def test_experiment_rejects_cutoff_before_test_end() -> None:
    experiment = _experiment_spec()

    with pytest.raises(ValueError, match="Data cutoff"):
        ExperimentSpec(
            name=experiment.name,
            version=experiment.version,
            feature_set=experiment.feature_set,
            target_set=experiment.target_set,
            task=experiment.task,
            universe=experiment.universe,
            data_cutoff=date(2024, 12, 31),
            split=experiment.split,
            model=experiment.model,
            random_seeds=experiment.random_seeds,
        )


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_experiment_rejects_invalid_random_seeds(seed: object) -> None:
    experiment = _experiment_spec()

    with pytest.raises(ValueError, match="Random seeds"):
        ExperimentSpec(
            name=experiment.name,
            version=experiment.version,
            feature_set=experiment.feature_set,
            target_set=experiment.target_set,
            task=experiment.task,
            universe=experiment.universe,
            data_cutoff=experiment.data_cutoff,
            split=experiment.split,
            model=experiment.model,
            random_seeds={"model": seed},  # type: ignore[dict-item]
        )


def test_model_rejects_non_json_configuration_values() -> None:
    with pytest.raises(TypeError, match="unsupported value"):
        ModelSpec(
            name="bad",
            version="1",
            model_type="example.Model",
            hyperparameters={"callback": object()},
        )


@pytest.mark.parametrize("value", ["", 1, None])
def test_universe_rejects_invalid_tickers(value: object) -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        UniverseSpec(
            name="universe",
            version="1",
            provider="yfinance",
            tickers=(value,),  # type: ignore[arg-type]
        )


def test_contract_names_must_be_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        UniverseSpec(
            name=1,  # type: ignore[arg-type]
            version="1",
            provider="yfinance",
            tickers=("ABB.ST",),
        )


def test_temporal_split_rejects_non_date_values() -> None:
    with pytest.raises(TypeError, match="datetime.date"):
        TemporalSplitSpec(
            name="invalid",
            version="1",
            train_start="2020-01-01",  # type: ignore[arg-type]
            train_end=date(2021, 12, 31),
            validation_start=date(2022, 1, 1),
            validation_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1),
            test_end=date(2025, 12, 31),
        )


def test_experiment_rejects_non_date_cutoff() -> None:
    experiment = _experiment_spec()

    with pytest.raises(TypeError, match="Data cutoff"):
        ExperimentSpec(
            name=experiment.name,
            version=experiment.version,
            feature_set=experiment.feature_set,
            target_set=experiment.target_set,
            task=experiment.task,
            universe=experiment.universe,
            data_cutoff="2025-12-31",  # type: ignore[arg-type]
            split=experiment.split,
            model=experiment.model,
            random_seeds=experiment.random_seeds,
        )


def test_universe_rejects_one_string_as_ticker_collection() -> None:
    with pytest.raises(TypeError, match="iterable of ticker strings"):
        UniverseSpec(
            name="universe",
            version="1",
            provider="yfinance",
            tickers="ABB.ST",  # type: ignore[arg-type]
        )


def test_model_hyperparameters_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        ModelSpec(
            name="model",
            version="1",
            model_type="example.Model",
            hyperparameters=[("depth", 3)],  # type: ignore[arg-type]
        )


def test_random_seeds_must_be_a_mapping() -> None:
    experiment = _experiment_spec()

    with pytest.raises(TypeError, match="provided as a mapping"):
        ExperimentSpec(
            name=experiment.name,
            version=experiment.version,
            feature_set=experiment.feature_set,
            target_set=experiment.target_set,
            task=experiment.task,
            universe=experiment.universe,
            data_cutoff=experiment.data_cutoff,
            split=experiment.split,
            model=experiment.model,
            random_seeds=[("model", 17)],  # type: ignore[arg-type]
        )


def test_universe_normalizes_ticker_whitespace_before_sorting() -> None:
    universe = UniverseSpec(
        name="universe",
        version="1",
        provider="yfinance",
        tickers=(" VOLV-B.ST ", "ABB.ST"),
    )

    assert universe.tickers == ("ABB.ST", "VOLV-B.ST")


def test_universe_rejects_duplicates_after_ticker_normalization() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        UniverseSpec(
            name="universe",
            version="1",
            provider="yfinance",
            tickers=("ABB.ST", " ABB.ST "),
        )
