from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

import swingtrader.modeling.training.baselines as baselines_module
import swingtrader.modeling.training.harness as harness_module
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
    TemporalCrossValidationSpec,
    TemporalSplitResult,
    TemporalSplitSpec,
    build_expanding_temporal_folds,
    resolve_model_feature_columns,
)
from swingtrader.modeling.training import (
    CONSTANT_PRIOR_MODEL_TYPE,
    LOGISTIC_REGRESSION_MODEL_TYPE,
    RANDOM_RANKING_MODEL_TYPE,
    TEMPORAL_CV_RESULT_COLUMNS,
    EvaluationConfig,
    RegularizedLogisticRegression,
    fit_baseline_model,
    run_baseline_cross_validation,
)


def _identity(data: pd.DataFrame) -> pd.DataFrame:
    return data


def _bundle_and_experiment() -> tuple[TemporalDatasetBundle, ExperimentSpec]:
    all_dates = pd.bdate_range("2020-01-01", periods=40)
    signal_dates = all_dates[:-2]
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", ticker, trading_date)
            for ticker in ("AAA.ST", "BBB.ST")
            for trading_date in signal_dates
        ],
        names=("provider", "ticker", "trading_date"),
    )
    date_number = pd.Series(
        pd.DatetimeIndex(index.get_level_values("trading_date")).map(
            {trading_date: position for position, trading_date in enumerate(all_dates)}
        ),
        index=index,
        dtype="float64",
    )
    ticker_signal = pd.Series(
        (index.get_level_values("ticker") == "BBB.ST").astype("float64"),
        index=index,
    )
    features = pd.DataFrame(
        {
            "feature_a": ticker_signal,
            "feature_b": date_number,
            "unused_feature": 10_000.0 + date_number,
        },
        index=index,
    )
    target = ticker_signal.astype("boolean")
    target_end_lookup = {
        signal_date: all_dates[position + 2]
        for position, signal_date in enumerate(signal_dates)
    }
    target_end = pd.DatetimeIndex(
        [target_end_lookup[trading_date] for trading_date in index.get_level_values("trading_date")]
    )
    targets = pd.DataFrame(
        {
            "target": target,
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
                output_columns=("feature_a", "feature_b", "unused_feature"),
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
                output_columns=("target", "target_resolution_date"),
                maximum_horizon_sessions=2,
            ),
        ),
    )
    task = SupervisedTaskSpec(
        name="binary",
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target",
        task_type="classification",
        horizon_sessions=2,
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
        data_cutoff=all_dates[-1].date(),
    )
    manifest = TemporalDatasetManifest(
        spec=dataset_spec,
        feature_columns=tuple(features.columns),
        target_columns=tuple(targets.columns),
        sample_columns=SAMPLE_METADATA_COLUMNS,
        source_row_count=len(index),
        sample_row_count=len(index),
        excluded_missing_target_count=0,
        observed_ticker_count=2,
        signal_date_start=signal_dates[0].date(),
        signal_date_end=signal_dates[-1].date(),
        target_end_date_start=all_dates[2].date(),
        target_end_date_end=all_dates[-1].date(),
        feature_missing_counts=tuple((column, 0) for column in features.columns),
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
        train_start=signal_dates[0].date(),
        train_end=signal_dates[23].date(),
        validation_start=signal_dates[24].date(),
        validation_end=signal_dates[29].date(),
        test_start=signal_dates[30].date(),
        test_end=signal_dates[-1].date(),
    )
    experiment = ExperimentSpec(
        name="logistic_cv",
        version="1",
        feature_set=feature_set,
        target_set=target_set,
        task=task,
        universe=universe,
        data_cutoff=dataset_spec.data_cutoff,
        split=split,
        model=ModelSpec(
            name="logistic",
            version="1",
            model_type=LOGISTIC_REGRESSION_MODEL_TYPE,
            hyperparameters={"regularization_strength": 0.1, "max_iter": 500},
            feature_columns=("feature_b", "feature_a"),
        ),
        random_seeds={"model": 17, "evaluation": 23},
    )
    return bundle, experiment


def _cv_spec() -> TemporalCrossValidationSpec:
    return TemporalCrossValidationSpec(
        n_folds=2,
        validation_sessions=4,
        minimum_train_sessions=8,
    )


def test_model_feature_column_resolution_preserves_declared_order() -> None:
    available = ("feature_a", "feature_b", "unused_feature")
    all_features = ModelSpec(name="all", version="1", model_type="example.Model")
    selected = ModelSpec(
        name="selected",
        version="1",
        model_type="example.Model",
        feature_columns=("feature_b", "feature_a"),
    )

    assert resolve_model_feature_columns(all_features, available) == available
    assert "feature_columns" not in all_features.to_manifest()
    assert resolve_model_feature_columns(selected, available) == ("feature_b", "feature_a")
    assert selected.to_manifest()["feature_columns"] == ["feature_b", "feature_a"]
    reordered = replace(selected, feature_columns=("feature_a", "feature_b"))
    assert selected.digest != reordered.digest


@pytest.mark.parametrize(
    ("feature_columns", "message"),
    [
        ((), "must not be empty"),
        (("feature_a", "feature_a"), "must be unique"),
        (("feature_a", 1), "non-empty strings"),
    ],
)
def test_model_spec_rejects_invalid_explicit_feature_columns(
    feature_columns: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ModelSpec(
            name="invalid",
            version="1",
            model_type="example.Model",
            feature_columns=feature_columns,  # type: ignore[arg-type]
        )


def test_model_feature_resolution_rejects_non_string_available_columns() -> None:
    model = ModelSpec(name="invalid_available", version="1", model_type="example.Model")

    with pytest.raises(ValueError, match="Available feature columns must be non-empty strings"):
        resolve_model_feature_columns(model, ("feature_a", 1))  # type: ignore[arg-type]


def test_model_feature_resolution_rejects_unknown_columns() -> None:
    model = ModelSpec(
        name="unknown",
        version="1",
        model_type="example.Model",
        feature_columns=("missing",),
    )
    with pytest.raises(ValueError, match="Unknown model feature columns: missing"):
        resolve_model_feature_columns(model, ("feature_a", "feature_b"))


@pytest.mark.parametrize(
    "model_type",
    (
        CONSTANT_PRIOR_MODEL_TYPE,
        RANDOM_RANKING_MODEL_TYPE,
        LOGISTIC_REGRESSION_MODEL_TYPE,
    ),
)
def test_all_feature_mode_retains_available_order_for_each_baseline(
    model_type: str,
) -> None:
    bundle, experiment = _bundle_and_experiment()
    features = bundle.features.iloc[:40]
    target = bundle.targets[experiment.task.target_column].iloc[:40]
    model_spec = ModelSpec(
        name="all_features",
        version="1",
        model_type=model_type,
        hyperparameters=(
            {"regularization_strength": 0.1, "max_iter": 500}
            if model_type == LOGISTIC_REGRESSION_MODEL_TYPE
            else {}
        ),
    )

    model = fit_baseline_model(
        model_spec,
        features=features,
        target=target,
        seed=17,
    )

    if model_type == LOGISTIC_REGRESSION_MODEL_TYPE:
        assert model.feature_columns == tuple(features.columns)
    else:
        assert model.feature_columns is None
    assert "feature_columns" not in model.to_manifest()


def test_explicit_feature_schema_is_retained_by_each_baseline() -> None:
    bundle, experiment = _bundle_and_experiment()
    features = bundle.features.iloc[:40]
    target = bundle.targets[experiment.task.target_column].iloc[:40]

    for model_type in (
        CONSTANT_PRIOR_MODEL_TYPE,
        RANDOM_RANKING_MODEL_TYPE,
        LOGISTIC_REGRESSION_MODEL_TYPE,
    ):
        model_spec = ModelSpec(
            name="selected",
            version="1",
            model_type=model_type,
            hyperparameters=(
                {"regularization_strength": 0.1, "max_iter": 500}
                if model_type == LOGISTIC_REGRESSION_MODEL_TYPE
                else {}
            ),
            feature_columns=("feature_b", "feature_a"),
        )
        model = fit_baseline_model(
            model_spec,
            features=features,
            target=target,
            seed=17,
        )

        assert model.feature_columns == ("feature_b", "feature_a")
        manifest = model.to_manifest()
        if model_type == LOGISTIC_REGRESSION_MODEL_TYPE:
            assert manifest["preprocessing"]["columns"] == ["feature_b", "feature_a"]
        else:
            assert manifest["feature_columns"] == ["feature_b", "feature_a"]
        with pytest.raises(ValueError, match="missing feature columns: feature_b"):
            model.predict_scores(features.loc[:, ["feature_a", "unused_feature"]])


def test_legacy_constant_and_random_artifact_construction_keeps_all_feature_behavior() -> None:
    bundle, _ = _bundle_and_experiment()
    features = bundle.features.iloc[:4]

    constant = baselines_module.ConstantPriorClassifier(prior=0.25, training_rows=10)
    random = baselines_module.DateMatchedRandomRanker(seed=17, training_rows=10)

    assert constant.feature_columns is None
    assert random.feature_columns is None
    assert constant.predict_scores(features).index.equals(features.index)
    assert random.predict_scores(features).index.equals(features.index)
    assert "feature_columns" not in constant.to_manifest()
    assert "feature_columns" not in random.to_manifest()


def test_logistic_fits_and_predicts_with_the_same_selected_schema() -> None:
    bundle, experiment = _bundle_and_experiment()
    features = bundle.features.iloc[:40]
    target = bundle.targets[experiment.task.target_column].iloc[:40]

    model = fit_baseline_model(
        experiment.model,
        features=features,
        target=target,
        seed=17,
    )

    assert isinstance(model, RegularizedLogisticRegression)
    assert model.feature_columns == ("feature_b", "feature_a")
    assert model.preprocessor.columns == ("feature_b", "feature_a")
    assert tuple(model.to_manifest()["coefficients"]) == ("feature_b", "feature_a")
    assert model.to_manifest()["preprocessing"]["columns"] == ["feature_b", "feature_a"]
    pd.testing.assert_series_equal(
        model.predict_scores(features),
        model.predict_scores(features.loc[:, ["feature_b", "feature_a"]]),
    )
    with pytest.raises(ValueError, match="missing feature columns: feature_b"):
        model.predict_scores(features.loc[:, ["feature_a", "unused_feature"]])


def test_expanding_folds_stay_inside_outer_train_and_apply_purging() -> None:
    bundle, experiment = _bundle_and_experiment()
    outer_split = FixedTemporalSplitter(experiment.split).assign(bundle)
    folds = build_expanding_temporal_folds(bundle, outer_split, spec=_cv_spec())
    repeated = build_expanding_temporal_folds(bundle, outer_split, spec=_cv_spec())
    outer_train = set(outer_split.indices("train"))
    forbidden = set(outer_split.indices("validation")) | set(outer_split.indices("test"))
    signal_dates = pd.DatetimeIndex(bundle.samples.index.get_level_values("trading_date"))
    target_end_dates = pd.DatetimeIndex(bundle.samples[TARGET_END_DATE_COLUMN])

    assert len(folds) == 2
    for fold, repeated_fold in zip(folds, repeated, strict=True):
        np.testing.assert_array_equal(fold.train_indices, repeated_fold.train_indices)
        np.testing.assert_array_equal(fold.validation_indices, repeated_fold.validation_indices)
        assert set(fold.train_indices).issubset(outer_train)
        assert set(fold.validation_indices).issubset(outer_train)
        assert set(fold.train_indices).isdisjoint(forbidden)
        assert set(fold.validation_indices).isdisjoint(forbidden)
        assert signal_dates.take(fold.train_indices).max().date() < fold.validation_start
        assert signal_dates.take(fold.validation_indices).min().date() >= fold.validation_start
        assert (target_end_dates.take(fold.train_indices).date <= fold.train_end).all()
        assert (target_end_dates.take(fold.validation_indices).date <= fold.validation_end).all()
        assert fold.train_end not in set(signal_dates.take(fold.train_indices).date)
        assert fold.validation_end not in set(signal_dates.take(fold.validation_indices).date)
        assert set(signal_dates.take(fold.train_indices)).isdisjoint(
            signal_dates.take(fold.validation_indices)
        )
    assert set(folds[0].train_indices).issubset(folds[1].train_indices)

    for fold in folds:
        for indices, partition_end in (
            (fold.train_indices, fold.train_end),
            (fold.validation_indices, fold.validation_end),
        ):
            selected_dates = signal_dates.take(indices).unique()
            for trading_date in selected_dates:
                expected = set(
                    np.flatnonzero(
                        (signal_dates == trading_date)
                        & np.isin(np.arange(len(signal_dates)), outer_split.indices("train"))
                        & (target_end_dates <= pd.Timestamp(partition_end))
                    )
                )
                observed = set(indices[signal_dates.take(indices) == trading_date])
                assert observed == expected


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("n_folds", 0, ValueError),
        ("validation_sessions", True, TypeError),
        ("minimum_train_sessions", -1, ValueError),
    ],
)
def test_cross_validation_spec_rejects_invalid_values(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    values: dict[str, object] = {
        "n_folds": 2,
        "validation_sessions": 4,
        "minimum_train_sessions": 8,
    }
    values[field] = value
    with pytest.raises(exception):
        TemporalCrossValidationSpec(**values)  # type: ignore[arg-type]


def test_cross_validation_fits_fresh_selected_models_without_outer_holdout_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, experiment = _bundle_and_experiment()
    outer_split = FixedTemporalSplitter(experiment.split).assign(bundle)
    folds = build_expanding_temporal_folds(bundle, outer_split, spec=_cv_spec())
    fitted: list[RegularizedLogisticRegression] = []
    fitted_indices: list[pd.Index] = []
    original_fit = harness_module.fit_baseline_model
    original_indices = TemporalSplitResult.indices
    requested_splits: list[str] = []

    def recording_fit(*args: object, **kwargs: object) -> RegularizedLogisticRegression:
        features = kwargs["features"]
        assert isinstance(features, pd.DataFrame)
        model = original_fit(*args, **kwargs)
        assert isinstance(model, RegularizedLogisticRegression)
        fitted.append(model)
        fitted_indices.append(features.index.copy())
        return model

    def train_only_indices(self: TemporalSplitResult, name: str) -> np.ndarray:
        requested_splits.append(name)
        if name != "train":
            raise AssertionError("Inner cross-validation accessed an outer holdout split.")
        return original_indices(self, name)  # type: ignore[arg-type]

    monkeypatch.setattr(harness_module, "fit_baseline_model", recording_fit)
    monkeypatch.setattr(TemporalSplitResult, "indices", train_only_indices)
    result = run_baseline_cross_validation(
        bundle,
        outer_split,
        experiment,
        _cv_spec(),
        evaluation_config=EvaluationConfig(random_seed=23),
    )

    assert tuple(result.columns) == TEMPORAL_CV_RESULT_COLUMNS
    assert len(result) == len(folds) == len(fitted)
    assert requested_splits == ["train"]
    assert len({id(model.preprocessor) for model in fitted}) == len(fitted)
    assert len({id(model.estimator) for model in fitted}) == len(fitted)
    for fold, model, fitted_index in zip(folds, fitted, fitted_indices, strict=True):
        pd.testing.assert_index_equal(fitted_index, bundle.features.iloc[fold.train_indices].index)
        assert model.feature_columns == ("feature_b", "feature_a")
        expected_medians = tuple(
            float(value)
            for value in bundle.features.iloc[fold.train_indices]
            .loc[:, ["feature_b", "feature_a"]]
            .median()
        )
        assert model.preprocessor.medians == expected_medians
    assert result[
        [
            "train_precision",
            "validation_precision",
            "train_recall",
            "validation_recall",
            "train_roc_auc",
            "validation_roc_auc",
        ]
    ].notna().all().all()


def test_cross_validation_rejects_non_logistic_baseline() -> None:
    bundle, experiment = _bundle_and_experiment()
    experiment = replace(
        experiment,
        model=ModelSpec(
            name="constant",
            version="1",
            model_type=CONSTANT_PRIOR_MODEL_TYPE,
        ),
    )
    outer_split = FixedTemporalSplitter(experiment.split).assign(bundle)

    with pytest.raises(ValueError, match="supports only LOGISTIC_REGRESSION_MODEL_TYPE"):
        run_baseline_cross_validation(bundle, outer_split, experiment, _cv_spec())
