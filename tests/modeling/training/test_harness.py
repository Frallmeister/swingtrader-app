from __future__ import annotations

from datetime import date
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features.contracts import FeatureBlockSpec, FeatureSetSpec
from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.specifications import TemporalDatasetSpec, UniverseSpec
from swingtrader.modeling.datasets.temporal import (
    SAMPLE_METADATA_COLUMNS,
    TARGET_END_DATE_COLUMN,
    TRAINING_ELIGIBILITY_REASONS_COLUMN,
    TRAINING_ELIGIBLE_COLUMN,
    TemporalDatasetBundle,
    TemporalDatasetManifest,
)
from swingtrader.modeling.experiments import (
    ExperimentSpec,
    FixedTemporalSplitter,
    ModelSpec,
    TemporalSplitSpec,
)
from swingtrader.modeling.experiments.tracking import ExperimentRun
from swingtrader.modeling.training import (
    CONSTANT_PRIOR_MODEL_TYPE,
    EvaluationConfig,
    run_baseline_experiment,
)


def _identity(data: pd.DataFrame) -> pd.DataFrame:
    return data


def _bundle_and_experiment() -> tuple[TemporalDatasetBundle, ExperimentSpec]:
    dates = pd.date_range("2021-12-27", "2022-01-14", freq="B")
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", ticker, trading_date)
            for ticker in ("AAA.ST", "BBB.ST")
            for trading_date in dates
        ],
        names=("provider", "ticker", "trading_date"),
    )
    ticker_offset = index.get_level_values("ticker").map({"AAA.ST": 0.0, "BBB.ST": 0.5})
    day = np.tile(np.arange(len(dates), dtype=float), 2)
    feature = day + ticker_offset
    target = ((day.astype(int) + (ticker_offset > 0).astype(int)) % 2).astype(bool)
    target_end = pd.DatetimeIndex(index.get_level_values("trading_date")) + pd.Timedelta(days=1)
    ranking_return = np.where(target, 0.02, -0.01)

    features = pd.DataFrame({"feature": feature}, index=index)
    targets = pd.DataFrame(
        {
            "target": pd.Series(target, index=index, dtype="boolean"),
            "forward_return_1d": ranking_return,
            "target_resolution_date": target_end,
        },
        index=index,
    )
    samples = pd.DataFrame(
        {
            TARGET_END_DATE_COLUMN: target_end,
            TRAINING_ELIGIBLE_COLUMN: pd.Series(True, index=index, dtype="boolean"),
            TRAINING_ELIGIBILITY_REASONS_COLUMN: pd.Series(
                [()] * len(index), index=index, dtype="object"
            ),
        },
        index=index,
    )
    feature_set = FeatureSetSpec(
        name="test_features",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="identity",
                builder=_identity,
                output_columns=("feature",),
            ),
        ),
    )
    target_set = TargetSetSpec(
        name="test_targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="identity",
                builder=_identity,
                output_columns=("target", "forward_return_1d", "target_resolution_date"),
                maximum_horizon_sessions=1,
            ),
        ),
    )
    task = SupervisedTaskSpec(
        name="binary",
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target",
        task_type="classification",
        horizon_sessions=1,
        target_end_date_column="target_resolution_date",
    )
    universe = UniverseSpec(
        name="test_universe",
        version="1",
        provider="yfinance",
        tickers=("AAA.ST", "BBB.ST"),
    )
    dataset_spec = TemporalDatasetSpec(
        feature_set=feature_set,
        target_set=target_set,
        task=task,
        universe=universe,
        data_start=date(2021, 12, 27),
        data_end=date(2022, 1, 15),
    )
    manifest = TemporalDatasetManifest(
        spec=dataset_spec,
        feature_columns=("feature",),
        target_columns=("target", "forward_return_1d", "target_resolution_date"),
        sample_columns=SAMPLE_METADATA_COLUMNS,
        source_row_count=len(index),
        sample_row_count=len(index),
        excluded_missing_target_count=0,
        observed_ticker_count=2,
        signal_date_start=date(2021, 12, 27),
        signal_date_end=date(2022, 1, 14),
        target_end_date_start=date(2021, 12, 28),
        target_end_date_end=date(2022, 1, 15),
        feature_missing_counts=(("feature", 0),),
        selected_target_summary=(
            ("class:false", int((~target).sum())),
            ("class:true", int(target.sum())),
        ),
        eligible_ticker_count=2,
        eligibility_failure_counts=(),
    )
    bundle = TemporalDatasetBundle(
        features=features,
        targets=targets,
        samples=samples,
        manifest=manifest,
    )
    split = TemporalSplitSpec(
        name="holdout",
        version="1",
        train_start=date(2021, 12, 27),
        train_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 3),
        validation_end=date(2022, 1, 7),
        test_start=date(2022, 1, 10),
        test_end=date(2022, 1, 15),
    )
    experiment = ExperimentSpec(
        name="baseline",
        version="1",
        feature_set=feature_set,
        target_set=target_set,
        task=task,
        universe=universe,
        data_start=dataset_spec.data_start,
        data_end=dataset_spec.data_end,
        split=split,
        model=ModelSpec(
            name="constant_prior",
            version="1",
            model_type=CONSTANT_PRIOR_MODEL_TYPE,
        ),
        random_seeds={"model": 17, "evaluation": 23},
    )
    return bundle, experiment


def test_harness_evaluates_validation_without_unlocking_test(tmp_path) -> None:
    bundle, experiment = _bundle_and_experiment()
    split_result = FixedTemporalSplitter(experiment.split).assign(bundle)
    result = run_baseline_experiment(
        bundle,
        split_result,
        experiment,
        ranking_return_column="forward_return_1d",
        evaluation_config=EvaluationConfig(top_k=1, random_seed=23),
        artifact_directory=tmp_path,
    )
    assert set(result.reports) == {"validation"}
    assert result.model.training_rows == len(split_result.indices("train"))
    assert (tmp_path / "model.json").is_file()
    assert (tmp_path / "validation" / "report.md").is_file()
    assert result.reports["validation"].ranking_return_column == "forward_return_1d"
    assert result.reports["validation"].to_manifest()["ranking_return_column"] == (
        "forward_return_1d"
    )


def test_harness_rejects_evaluation_seed_outside_the_experiment_contract() -> None:
    bundle, experiment = _bundle_and_experiment()
    split_result = FixedTemporalSplitter(experiment.split).assign(bundle)

    with pytest.raises(ValueError, match="must match the experiment evaluation seed"):
        run_baseline_experiment(
            bundle,
            split_result,
            experiment,
            evaluation_config=EvaluationConfig(random_seed=99),
        )


def test_locked_test_requires_explicit_opt_in() -> None:
    bundle, experiment = _bundle_and_experiment()
    split_result = FixedTemporalSplitter(experiment.split).assign(bundle)
    validation_only = run_baseline_experiment(bundle, split_result, experiment)
    result = run_baseline_experiment(
        bundle,
        split_result,
        experiment,
        include_locked_test=True,
    )
    assert set(result.reports) == {"validation", "test"}
    pd.testing.assert_frame_equal(
        validation_only.reports["validation"].predictions,
        result.reports["validation"].predictions,
    )


def test_harness_logs_finite_metrics_and_generated_artifacts() -> None:
    bundle, experiment = _bundle_and_experiment()
    split_result = FixedTemporalSplitter(experiment.split).assign(bundle)
    captured_metrics: list[dict[str, float]] = []
    captured_artifacts: list[tuple[str, str | None]] = []
    mlflow = ModuleType("fake_mlflow")
    mlflow.log_metrics = lambda metrics, step=None: captured_metrics.append(dict(metrics))
    mlflow.log_artifact = lambda path, artifact_path=None: captured_artifacts.append(
        (path, artifact_path)
    )
    run = ExperimentRun(mlflow, "run-id")
    run_baseline_experiment(bundle, split_result, experiment, run=run)
    assert captured_metrics
    assert all(np.isfinite(value) for value in captured_metrics[0].values())
    assert any(path.endswith("model.json") for path, _ in captured_artifacts)
    assert any(path.endswith("predictions.csv.gz") for path, _ in captured_artifacts)
