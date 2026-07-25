from __future__ import annotations

import json
from datetime import date

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
from swingtrader.modeling.experiments.contracts import TemporalSplitSpec
from swingtrader.modeling.experiments.splitting import (
    EMBARGO_REASON,
    OUTSIDE_SPLIT_RANGES_REASON,
    SPLIT_COLUMN,
    SPLIT_EXCLUSION_REASON_COLUMN,
    TemporalSplitResult,
    TARGET_END_AFTER_SPLIT_REASON,
    FixedTemporalSplitter,
    split_temporal_dataset,
)


def _identity(data: pd.DataFrame) -> pd.DataFrame:
    return data


def _bundle() -> TemporalDatasetBundle:
    rows = [
        ("yfinance", "AAA.ST", "2021-12-28", "2021-12-30", False),
        ("yfinance", "AAA.ST", "2021-12-29", "2022-01-05", True),
        ("yfinance", "AAA.ST", "2021-12-30", "2021-12-31", True),
        ("yfinance", "AAA.ST", "2022-01-03", "2022-01-04", False),
        ("yfinance", "AAA.ST", "2022-01-05", "2022-01-07", True),
        ("yfinance", "AAA.ST", "2022-01-06", "2022-01-10", True),
        ("yfinance", "AAA.ST", "2022-01-10", "2022-01-12", False),
        ("yfinance", "AAA.ST", "2022-01-13", "2022-01-17", True),
        ("yfinance", "AAA.ST", "2022-01-15", "2022-01-16", False),
        ("yfinance", "BBB.ST", "2021-12-27", "2021-12-29", False),
        ("yfinance", "BBB.ST", "2021-12-30", "2021-12-31", True),
        ("yfinance", "BBB.ST", "2022-01-04", "2022-01-06", False),
        ("yfinance", "BBB.ST", "2022-01-05", "2022-01-07", True),
        ("yfinance", "BBB.ST", "2022-01-11", "2022-01-13", False),
        ("yfinance", "BBB.ST", "2022-01-12", "2022-01-14", True),
        ("yfinance", "BBB.ST", "2022-01-15", "2022-01-16", True),
    ]
    index = pd.MultiIndex.from_tuples(
        [
            (provider, ticker, pd.Timestamp(signal_date))
            for provider, ticker, signal_date, _, _ in rows
        ],
        names=("provider", "ticker", "trading_date"),
    )
    target_end_dates = pd.to_datetime([target_end_date for *_, target_end_date, _ in rows])
    target = pd.Series([value for *_, value in rows], index=index, dtype="boolean")
    features = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    targets = pd.DataFrame({"target": target}, index=index)
    samples = pd.DataFrame(
        {
            TARGET_END_DATE_COLUMN: target_end_dates,
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
                output_columns=("target",),
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
    )
    spec = TemporalDatasetSpec(
        feature_set=feature_set,
        target_set=target_set,
        task=task,
        universe=UniverseSpec(
            name="test_universe",
            version="1",
            provider="yfinance",
            tickers=("AAA.ST", "BBB.ST"),
        ),
        data_cutoff=date(2022, 1, 20),
    )
    manifest = TemporalDatasetManifest(
        spec=spec,
        feature_columns=("feature",),
        target_columns=("target",),
        sample_columns=SAMPLE_METADATA_COLUMNS,
        source_row_count=len(index),
        sample_row_count=len(index),
        excluded_missing_target_count=0,
        observed_ticker_count=2,
        signal_date_start=date(2021, 12, 27),
        signal_date_end=date(2022, 1, 15),
        target_end_date_start=date(2021, 12, 29),
        target_end_date_end=date(2022, 1, 17),
        feature_missing_counts=(("feature", 0),),
        selected_target_summary=(
            ("class:false", int((~target).sum())),
            ("class:true", int(target.sum())),
        ),
        eligible_ticker_count=2,
        eligibility_failure_counts=(),
    )
    return TemporalDatasetBundle(
        features=features,
        targets=targets,
        samples=samples,
        manifest=manifest,
    )


def _split_spec(*, embargo_sessions: int = 0) -> TemporalSplitSpec:
    return TemporalSplitSpec(
        name="initial_holdout",
        version="1",
        train_start=date(2021, 12, 27),
        train_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2022, 1, 7),
        test_start=date(2022, 1, 8),
        test_end=date(2022, 1, 14),
        embargo_sessions=embargo_sessions,
    )


def _row(result: TemporalSplitResult, ticker: str, signal_date: str) -> pd.Series:
    return result.samples.loc[("yfinance", ticker, pd.Timestamp(signal_date))]


def test_split_purges_using_each_rows_actual_target_end_date() -> None:
    bundle = _bundle()
    result = split_temporal_dataset(bundle, spec=_split_spec())

    assert _row(result, "AAA.ST", "2021-12-29")[SPLIT_COLUMN] is pd.NA
    assert (
        _row(result, "AAA.ST", "2021-12-29")[SPLIT_EXCLUSION_REASON_COLUMN]
        == TARGET_END_AFTER_SPLIT_REASON
    )
    assert (
        _row(result, "AAA.ST", "2022-01-06")[SPLIT_EXCLUSION_REASON_COLUMN]
        == TARGET_END_AFTER_SPLIT_REASON
    )
    assert (
        _row(result, "AAA.ST", "2022-01-13")[SPLIT_EXCLUSION_REASON_COLUMN]
        == TARGET_END_AFTER_SPLIT_REASON
    )

    ends = {
        "train": _split_spec().train_end,
        "validation": _split_spec().validation_end,
        "test": _split_spec().test_end,
    }
    for split_name, split_end in ends.items():
        assigned = result.samples[SPLIT_COLUMN].eq(split_name).fillna(False)
        assert (
            pd.DatetimeIndex(result.samples.loc[assigned, TARGET_END_DATE_COLUMN]).date
            <= split_end
        ).all()


def test_target_end_on_inclusive_split_end_is_retained() -> None:
    result = split_temporal_dataset(_bundle(), spec=_split_spec())

    assert _row(result, "AAA.ST", "2021-12-30")[SPLIT_COLUMN] == "train"
    assert _row(result, "BBB.ST", "2022-01-05")[SPLIT_COLUMN] == "validation"
    assert _row(result, "BBB.ST", "2022-01-12")[SPLIT_COLUMN] == "test"


def test_common_calendar_boundaries_apply_to_different_ticker_histories() -> None:
    result = split_temporal_dataset(_bundle(), spec=_split_spec())

    assert _row(result, "AAA.ST", "2022-01-03")[SPLIT_COLUMN] == "validation"
    assert _row(result, "BBB.ST", "2022-01-04")[SPLIT_COLUMN] == "validation"
    assert _row(result, "AAA.ST", "2022-01-10")[SPLIT_COLUMN] == "test"
    assert _row(result, "BBB.ST", "2022-01-11")[SPLIT_COLUMN] == "test"


def test_embargo_removes_additional_global_dates_after_purging() -> None:
    result = split_temporal_dataset(_bundle(), spec=_split_spec(embargo_sessions=1))

    train_embargo = result.samples.index.get_level_values("trading_date") == pd.Timestamp(
        "2021-12-30"
    )
    validation_embargo = result.samples.index.get_level_values("trading_date") == pd.Timestamp(
        "2022-01-05"
    )
    assert set(result.samples.loc[train_embargo, SPLIT_EXCLUSION_REASON_COLUMN]) == {
        EMBARGO_REASON
    }
    assert set(result.samples.loc[validation_embargo, SPLIT_EXCLUSION_REASON_COLUMN]) == {
        EMBARGO_REASON
    }
    assert set(result.samples.loc[train_embargo].index.get_level_values("ticker")) == {
        "AAA.ST",
        "BBB.ST",
    }
    assert (
        _row(result, "AAA.ST", "2021-12-29")[SPLIT_EXCLUSION_REASON_COLUMN]
        == TARGET_END_AFTER_SPLIT_REASON
    )
    assert result.summary("test").embargoed_row_count == 0


def test_outside_rows_remain_in_metadata_with_an_explicit_reason() -> None:
    result = split_temporal_dataset(_bundle(), spec=_split_spec())

    for ticker in ("AAA.ST", "BBB.ST"):
        row = _row(result, ticker, "2022-01-15")
        assert row[SPLIT_COLUMN] is pd.NA
        assert row[SPLIT_EXCLUSION_REASON_COLUMN] == OUTSIDE_SPLIT_RANGES_REASON


def test_split_manifest_reports_counts_dates_tickers_and_prevalence() -> None:
    result = split_temporal_dataset(_bundle(), spec=_split_spec(embargo_sessions=1))
    manifest = result.manifest

    assert manifest.source_row_count == 16
    assert manifest.outside_range_row_count == 2
    assert manifest.purged_row_count == 3
    assert manifest.embargoed_row_count == 4
    assert manifest.assigned_row_count == 7
    assert manifest.summary("train").trading_date_count == 2
    assert manifest.summary("train").ticker_count == 2
    assert manifest.summary("train").signal_date_end == date(2021, 12, 28)
    assert manifest.summary("validation").signal_date_end == date(2022, 1, 4)
    assert manifest.summary("test").class_prevalence == pytest.approx(1 / 3)
    assert json.loads(manifest.to_json()) == manifest.to_manifest()


def test_split_assignment_is_deterministic_and_does_not_mutate_bundle() -> None:
    bundle = _bundle()
    original_samples = bundle.samples.copy(deep=True)

    first = split_temporal_dataset(bundle, spec=_split_spec(embargo_sessions=1))
    second = split_temporal_dataset(bundle, spec=_split_spec(embargo_sessions=1))

    pd.testing.assert_frame_equal(first.samples, second.samples)
    pd.testing.assert_frame_equal(bundle.samples, original_samples)
    assert first.manifest.to_manifest() == second.manifest.to_manifest()
    assert first.manifest.digest == second.manifest.digest


def test_scikit_style_iteration_excludes_locked_test_indices() -> None:
    bundle = _bundle()
    splitter = FixedTemporalSplitter(_split_spec())
    result = splitter.assign(bundle)

    [(train_indices, validation_indices)] = list(splitter.split(bundle))

    np.testing.assert_array_equal(train_indices, result.indices("train"))
    np.testing.assert_array_equal(validation_indices, result.indices("validation"))
    assert set(train_indices).isdisjoint(result.indices("test"))
    assert set(validation_indices).isdisjoint(result.indices("test"))
    assert splitter.get_n_splits() == 1


def test_split_positions_align_all_canonical_bundle_frames() -> None:
    bundle = _bundle()
    result = split_temporal_dataset(bundle, spec=_split_spec())

    for name in ("train", "validation", "test"):
        positions = result.indices(name)
        expected_index = result.sample_index(name)
        pd.testing.assert_index_equal(bundle.features.iloc[positions].index, expected_index)
        pd.testing.assert_index_equal(bundle.targets.iloc[positions].index, expected_index)
        pd.testing.assert_index_equal(bundle.samples.iloc[positions].index, expected_index)

    pd.testing.assert_series_equal(
        result.samples[TRAINING_ELIGIBLE_COLUMN],
        bundle.samples[TRAINING_ELIGIBLE_COLUMN],
    )
    pd.testing.assert_series_equal(
        result.samples[TRAINING_ELIGIBILITY_REASONS_COLUMN],
        bundle.samples[TRAINING_ELIGIBILITY_REASONS_COLUMN],
    )


def test_fixed_splitter_rejects_an_invalid_spec_at_construction() -> None:
    with pytest.raises(TypeError, match="TemporalSplitSpec"):
        FixedTemporalSplitter(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("embargo_sessions", [-1, True, 1.5])
def test_split_spec_rejects_invalid_embargo(embargo_sessions: object) -> None:
    error = TypeError if embargo_sessions is True or embargo_sessions == 1.5 else ValueError
    with pytest.raises(error, match="embargo"):
        _split_spec(embargo_sessions=embargo_sessions)  # type: ignore[arg-type]


def test_split_spec_manifest_records_embargo_and_changes_digest() -> None:
    without_embargo = _split_spec()
    with_embargo = _split_spec(embargo_sessions=1)

    assert without_embargo.to_manifest()["embargo_sessions"] == 0
    assert with_embargo.to_manifest()["embargo_sessions"] == 1
    assert with_embargo.digest != without_embargo.digest


def test_rows_in_calendar_gaps_are_outside_and_crossing_targets_are_still_purged() -> None:
    spec = TemporalSplitSpec(
        name="gapped_holdout",
        version="1",
        train_start=date(2021, 12, 27),
        train_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 6),
        validation_end=date(2022, 1, 10),
        test_start=date(2022, 1, 11),
        test_end=date(2022, 1, 14),
    )

    result = split_temporal_dataset(_bundle(), spec=spec)

    assert (
        _row(result, "AAA.ST", "2021-12-29")[SPLIT_EXCLUSION_REASON_COLUMN]
        == TARGET_END_AFTER_SPLIT_REASON
    )
    assert (
        _row(result, "AAA.ST", "2022-01-03")[SPLIT_EXCLUSION_REASON_COLUMN]
        == OUTSIDE_SPLIT_RANGES_REASON
    )
    assert (
        _row(result, "BBB.ST", "2022-01-05")[SPLIT_EXCLUSION_REASON_COLUMN]
        == OUTSIDE_SPLIT_RANGES_REASON
    )


def test_split_rejects_test_end_after_dataset_cutoff() -> None:
    spec = TemporalSplitSpec(
        name="invalid",
        version="1",
        train_start=date(2021, 12, 27),
        train_end=date(2021, 12, 31),
        validation_start=date(2022, 1, 1),
        validation_end=date(2022, 1, 7),
        test_start=date(2022, 1, 8),
        test_end=date(2022, 1, 21),
    )

    with pytest.raises(ValueError, match="dataset cutoff"):
        split_temporal_dataset(_bundle(), spec=spec)


def test_split_rejects_an_embargo_that_empties_a_split() -> None:
    with pytest.raises(ValueError, match="too few surviving signal dates"):
        split_temporal_dataset(_bundle(), spec=_split_spec(embargo_sessions=3))
