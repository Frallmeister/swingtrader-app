import sys
from datetime import date, datetime
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine, insert

from swingtrader.data.bronze.schema import bronze_market_daily_prices, metadata
from swingtrader.data.features.contracts import (
    FeatureBlockSpec,
    FeatureSetSpec,
    HistoryRequirement,
)
from swingtrader.modeling.datasets.contracts import (
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
)
from swingtrader.modeling.datasets.specifications import TemporalDatasetSpec, UniverseSpec
from swingtrader.modeling.datasets.tabular import to_tabular_dataset
from swingtrader.modeling.datasets.temporal import (
    TickerEligibility,
    build_temporal_dataset,
    construct_temporal_dataset,
)


def add_expanding_feature(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["feature_expanding_mean"] = result["adjusted_close"].groupby(
        level=["provider", "ticker"], sort=False
    ).transform(lambda values: values.expanding(min_periods=2).mean())
    return result


def add_two_session_target(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    future = result["adjusted_close"].groupby(
        level=["provider", "ticker"], sort=False
    ).shift(-2)
    target = pd.Series(pd.NA, index=result.index, dtype="boolean")
    valid = future.notna()
    target.loc[valid] = future.loc[valid].gt(result.loc[valid, "adjusted_close"])
    result["target_up_2d"] = target
    return result


def mutate_feature_index_in_place(data: pd.DataFrame) -> pd.DataFrame:
    data.index = data.index[::-1]
    data["feature_expanding_mean"] = 1.0
    return data


def add_event_target(data: pd.DataFrame) -> pd.DataFrame:
    result = add_two_session_target(data)
    dates = pd.Series(
        pd.DatetimeIndex(result.index.get_level_values("trading_date")),
        index=result.index,
    )
    result["target_end_date_2d"] = dates.groupby(
        level=["provider", "ticker"], sort=False
    ).shift(-1)
    return result


def test_build_temporal_dataset_loads_required_history_through_the_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def check_training_eligibility(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        states = tuple(
            SimpleNamespace(
                ticker=ticker,
                status="not_eligible",
                failure_reasons=(SimpleNamespace(value="insufficient_history"),),
            )
            for ticker in ("AAA", "BBB")
        )
        return SimpleNamespace(states=states)

    eligibility_module = ModuleType("swingtrader.data.eligibility")
    eligibility_module.TrainingEligibilityStatus = SimpleNamespace(ELIGIBLE="eligible")
    eligibility_module.check_training_eligibility = check_training_eligibility
    monkeypatch.setitem(sys.modules, "swingtrader.data.eligibility", eligibility_module)

    engine = create_engine("sqlite+pysqlite:///:memory:")
    metadata.create_all(engine)
    rows = [
        _bronze_row(ticker, trading_date, close=offset + position + 1.0)
        for ticker, offset in (("AAA", 0.0), ("BBB", 100.0))
        for position, trading_date in enumerate(pd.date_range("2026-01-01", periods=7))
    ]
    with engine.begin() as connection:
        connection.execute(insert(bronze_market_daily_prices), rows)

    bundle = build_temporal_dataset(
        engine=engine,
        spec=_spec(data_cutoff=date(2026, 1, 6)),
    )

    assert captured["data_cutoff"] == date(2026, 1, 6)
    assert bundle.manifest.source_row_count == 12
    assert bundle.manifest.sample_row_count == 8
    assert bundle.samples["training_eligible_at_cutoff"].eq(False).all()
    assert bundle.samples.index.get_level_values("trading_date").max() == pd.Timestamp(
        "2026-01-04"
    )


def test_construct_temporal_dataset_aligns_frames_and_preserves_feature_nans() -> None:
    prices = _prices(periods=6)
    original = prices.copy(deep=True)

    bundle = construct_temporal_dataset(
        prices,
        spec=_spec(data_cutoff=date(2026, 1, 6)),
        eligibility=_eligibility(),
    )

    assert bundle.features.index.equals(bundle.targets.index)
    assert bundle.features.index.equals(bundle.samples.index)
    assert list(bundle.features.columns) == ["feature_expanding_mean"]
    assert list(bundle.targets.columns) == ["target_up_2d"]
    assert len(bundle.features) == 8
    assert bundle.features["feature_expanding_mean"].isna().sum() == 2
    assert not bundle.targets["target_up_2d"].isna().any()
    assert bundle.manifest.excluded_missing_target_count == 4
    assert bundle.manifest.eligible_ticker_count == 1
    assert bundle.samples.loc[("test", "BBB"), "training_eligible_at_cutoff"].eq(False).all()
    pd.testing.assert_frame_equal(prices, original)


def test_source_row_order_does_not_affect_the_bundle() -> None:
    prices = _prices(periods=6)
    shuffled = prices.sample(frac=1.0, random_state=17)
    spec = _spec(data_cutoff=date(2026, 1, 6))

    ordered = construct_temporal_dataset(
        prices,
        spec=spec,
        eligibility=_eligibility(),
    )
    reordered = construct_temporal_dataset(
        shuffled,
        spec=spec,
        eligibility=_eligibility(),
    )

    pd.testing.assert_frame_equal(reordered.features, ordered.features)
    pd.testing.assert_frame_equal(reordered.targets, ordered.targets)
    pd.testing.assert_frame_equal(reordered.samples, ordered.samples)
    assert reordered.manifest.digest == ordered.manifest.digest


def test_duplicate_sample_keys_are_rejected() -> None:
    prices = _prices(periods=6)
    duplicate = pd.concat([prices, prices.iloc[[0]]])

    with pytest.raises(ValueError, match="unique index"):
        construct_temporal_dataset(
            duplicate,
            spec=_spec(data_cutoff=date(2026, 1, 6)),
            eligibility=_eligibility(),
        )


def test_changing_one_ticker_does_not_affect_another_ticker() -> None:
    baseline_prices = _prices(periods=6)
    changed_prices = baseline_prices.copy()
    changed_prices.loc[("test", "BBB"), "adjusted_close"] *= 10
    spec = _spec(data_cutoff=date(2026, 1, 6))

    baseline = construct_temporal_dataset(
        baseline_prices,
        spec=spec,
        eligibility=_eligibility(),
    )
    changed = construct_temporal_dataset(
        changed_prices,
        spec=spec,
        eligibility=_eligibility(),
    )

    pd.testing.assert_frame_equal(
        changed.features.loc[("test", "AAA")],
        baseline.features.loc[("test", "AAA")],
    )
    pd.testing.assert_frame_equal(
        changed.targets.loc[("test", "AAA")],
        baseline.targets.loc[("test", "AAA")],
    )


def test_feature_blocks_cannot_mutate_the_index_in_place() -> None:
    feature_set = FeatureSetSpec(
        name="mutating_features",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="mutating",
                builder=mutate_feature_index_in_place,
                output_columns=("feature_expanding_mean",),
                history_requirement=HistoryRequirement.PATH_DEPENDENT,
            ),
        ),
    )
    spec = _spec(data_cutoff=date(2026, 1, 6))
    spec = TemporalDatasetSpec(
        feature_set=feature_set,
        target_set=spec.target_set,
        task=spec.task,
        universe=spec.universe,
        data_cutoff=spec.data_cutoff,
    )

    with pytest.raises(ValueError, match="changed the canonical sample index"):
        construct_temporal_dataset(
            _prices(periods=6),
            spec=spec,
            eligibility=_eligibility(),
        )


def test_fixed_horizon_target_end_dates_use_observed_sessions() -> None:
    bundle = construct_temporal_dataset(
        _prices(periods=6),
        spec=_spec(data_cutoff=date(2026, 1, 6)),
        eligibility=_eligibility(),
    )

    first_index = ("test", "AAA", pd.Timestamp("2026-01-01"))
    assert bundle.samples.loc[first_index, "target_end_date"] == pd.Timestamp("2026-01-03")


def test_explicit_target_end_dates_are_used_for_event_targets() -> None:
    target_set = TargetSetSpec(
        name="event_targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="event",
                builder=add_event_target,
                required_columns=frozenset({"adjusted_close"}),
                output_columns=("target_up_2d", "target_end_date_2d"),
                maximum_horizon_sessions=2,
            ),
        ),
    )
    task = SupervisedTaskSpec(
        name="event_task",
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target_up_2d",
        task_type="classification",
        horizon_sessions=2,
        target_end_date_column="target_end_date_2d",
    )
    spec = _spec(
        data_cutoff=date(2026, 1, 6),
        target_set=target_set,
        task=task,
    )

    bundle = construct_temporal_dataset(
        _prices(periods=6),
        spec=spec,
        eligibility=_eligibility(),
    )

    first_index = ("test", "AAA", pd.Timestamp("2026-01-01"))
    assert bundle.samples.loc[first_index, "target_end_date"] == pd.Timestamp("2026-01-02")


def test_extending_the_cutoff_does_not_change_existing_feature_values() -> None:
    short = construct_temporal_dataset(
        _prices(periods=6),
        spec=_spec(data_cutoff=date(2026, 1, 6)),
        eligibility=_eligibility(),
    )
    long = construct_temporal_dataset(
        _prices(periods=8),
        spec=_spec(data_cutoff=date(2026, 1, 8)),
        eligibility=_eligibility(),
    )

    pd.testing.assert_frame_equal(long.features.loc[short.features.index], short.features)
    pd.testing.assert_frame_equal(long.targets.loc[short.targets.index], short.targets)
    assert len(long.targets) > len(short.targets)


def test_tabular_adapter_preserves_missing_values_and_metadata() -> None:
    bundle = construct_temporal_dataset(
        _prices(periods=6),
        spec=_spec(data_cutoff=date(2026, 1, 6)),
        eligibility=_eligibility(),
    )

    tabular = to_tabular_dataset(bundle)

    assert tabular.X.isna().sum().sum() == 2
    assert tabular.y.name == "target_up_2d"
    assert tabular.X.index.equals(tabular.samples.index)
    pd.testing.assert_frame_equal(tabular.samples, bundle.samples)


def test_manifest_is_deterministic_and_excludes_downstream_concepts() -> None:
    spec = _spec(data_cutoff=date(2026, 1, 6))
    first = construct_temporal_dataset(
        _prices(periods=6), spec=spec, eligibility=_eligibility()
    )
    second = construct_temporal_dataset(
        _prices(periods=6), spec=spec, eligibility=_eligibility()
    )

    assert first.manifest.digest == second.manifest.digest
    manifest_text = first.manifest.to_json()
    for excluded in ("split", "model", "random_seed", "mlflow"):
        assert excluded not in manifest_text.lower()


def test_regression_manifest_uses_compact_numeric_summary() -> None:
    def add_regression_target(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["target_return_2d"] = result["adjusted_close"].groupby(
            level=["provider", "ticker"], sort=False
        ).shift(-2) / result["adjusted_close"] - 1
        return result

    target_set = TargetSetSpec(
        name="regression_targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="return",
                builder=add_regression_target,
                required_columns=frozenset({"adjusted_close"}),
                output_columns=("target_return_2d",),
                maximum_horizon_sessions=2,
            ),
        ),
    )
    task = SupervisedTaskSpec(
        name="regression_task",
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target_return_2d",
        task_type="regression",
        horizon_sessions=2,
    )

    bundle = construct_temporal_dataset(
        _prices(periods=6),
        spec=_spec(
            data_cutoff=date(2026, 1, 6),
            target_set=target_set,
            task=task,
        ),
        eligibility=_eligibility(),
    )

    assert dict(bundle.manifest.selected_target_summary).keys() == {
        "count",
        "mean",
        "standard_deviation",
        "minimum",
        "maximum",
    }


def test_regression_manifest_rejects_non_finite_targets() -> None:
    def add_non_finite_target(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["target_return_2d"] = result["adjusted_close"].groupby(
            level=["provider", "ticker"], sort=False
        ).shift(-2) / result["adjusted_close"] - 1
        result.iloc[0, result.columns.get_loc("target_return_2d")] = float("inf")
        return result

    target_set = TargetSetSpec(
        name="non_finite_targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="return",
                builder=add_non_finite_target,
                required_columns=frozenset({"adjusted_close"}),
                output_columns=("target_return_2d",),
                maximum_horizon_sessions=2,
            ),
        ),
    )
    task = SupervisedTaskSpec(
        name="regression_task",
        target_set_name=target_set.name,
        target_set_version=target_set.version,
        target_column="target_return_2d",
        task_type="regression",
        horizon_sessions=2,
    )

    with pytest.raises(ValueError, match="finite values"):
        construct_temporal_dataset(
            _prices(periods=6),
            spec=_spec(
                data_cutoff=date(2026, 1, 6),
                target_set=target_set,
                task=task,
            ),
            eligibility=_eligibility(),
        )


def test_source_scope_requires_exact_universe_and_eligibility_membership() -> None:
    spec = _spec(data_cutoff=date(2026, 1, 6))

    with pytest.raises(ValueError, match="missing universe tickers"):
        construct_temporal_dataset(
            _prices(periods=6).loc[("test", "AAA") : ("test", "AAA")],
            spec=spec,
            eligibility=_eligibility(),
        )

    with pytest.raises(ValueError, match="cover exactly"):
        construct_temporal_dataset(
            _prices(periods=6),
            spec=spec,
            eligibility={"AAA": _eligibility()["AAA"]},
        )


def test_source_rows_after_the_cutoff_are_rejected() -> None:
    with pytest.raises(ValueError, match="after the dataset cutoff"):
        construct_temporal_dataset(
            _prices(periods=7),
            spec=_spec(data_cutoff=date(2026, 1, 6)),
            eligibility=_eligibility(),
        )


def test_empty_selected_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="No rows have an available"):
        construct_temporal_dataset(
            _prices(periods=2),
            spec=_spec(data_cutoff=date(2026, 1, 2)),
            eligibility=_eligibility(),
        )


def _bronze_row(
    ticker: str,
    trading_date: pd.Timestamp,
    *,
    close: float,
) -> dict[str, object]:
    return {
        "provider": "test",
        "ticker": ticker,
        "trading_date": trading_date.date(),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adjusted_close": close,
        "volume": 100,
        "dividends": 0,
        "stock_splits": 0,
        "fetched_at": datetime(2026, 1, 8),
        "request_id": "test-request",
    }


def _prices(*, periods: int) -> pd.DataFrame:
    rows: list[tuple[str, str, pd.Timestamp, float]] = []
    for ticker, offset in (("AAA", 0.0), ("BBB", 100.0)):
        for position, trading_date in enumerate(pd.date_range("2026-01-01", periods=periods)):
            rows.append(("test", ticker, trading_date, offset + position + 1.0))
    return (
        pd.DataFrame(
            rows,
            columns=["provider", "ticker", "trading_date", "adjusted_close"],
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )


def _spec(
    *,
    data_cutoff: date,
    target_set: TargetSetSpec | None = None,
    task: SupervisedTaskSpec | None = None,
) -> TemporalDatasetSpec:
    feature_set = FeatureSetSpec(
        name="test_features",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="expanding",
                builder=add_expanding_feature,
                required_columns=frozenset({"adjusted_close"}),
                output_columns=("feature_expanding_mean",),
                history_requirement=HistoryRequirement.EXPANDING,
            ),
        ),
    )
    resolved_target_set = target_set or TargetSetSpec(
        name="test_targets",
        version="1",
        families=(
            TargetFamilySpec(
                name="two_session",
                builder=add_two_session_target,
                required_columns=frozenset({"adjusted_close"}),
                output_columns=("target_up_2d",),
                maximum_horizon_sessions=2,
            ),
        ),
    )
    resolved_task = task or SupervisedTaskSpec(
        name="test_task",
        target_set_name=resolved_target_set.name,
        target_set_version=resolved_target_set.version,
        target_column="target_up_2d",
        task_type="classification",
        horizon_sessions=2,
    )
    return TemporalDatasetSpec(
        feature_set=feature_set,
        target_set=resolved_target_set,
        task=resolved_task,
        universe=UniverseSpec(
            name="test_universe",
            version="1",
            provider="test",
            tickers=("BBB", "AAA"),
        ),
        data_cutoff=data_cutoff,
    )


def _eligibility() -> dict[str, TickerEligibility]:
    return {
        "AAA": TickerEligibility(ticker="AAA", eligible=True),
        "BBB": TickerEligibility(
            ticker="BBB",
            eligible=False,
            failure_reasons=("insufficient_history",),
        ),
    }
