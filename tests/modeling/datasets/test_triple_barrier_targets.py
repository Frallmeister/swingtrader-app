from __future__ import annotations

import pandas as pd
import pytest

from swingtrader.modeling.datasets import (
    V3_PRIMARY_TASK,
    V3_TARGET_SET,
    add_triple_barrier_targets,
    generate_v3_labels,
)
from swingtrader.modeling.datasets.contracts import TargetFamilySpec, TargetSetSpec


def _prices(
    future_bars: list[tuple[float, float, float, float]],
    *,
    ticker: str = "AAA",
    provider: str = "test",
) -> pd.DataFrame:
    bars = [(100.0, 101.0, 99.0, 100.0), *future_bars]
    dates = pd.date_range("2026-01-02", periods=len(bars), freq="B")
    return (
        pd.DataFrame(
            {
                "provider": provider,
                "ticker": ticker,
                "trading_date": dates,
                "open": [bar[0] for bar in bars],
                "high": [bar[1] for bar in bars],
                "low": [bar[2] for bar in bars],
                "close": [bar[3] for bar in bars],
                "adjusted_close": [bar[3] for bar in bars],
            }
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )


def _add_targets(
    prices: pd.DataFrame,
    *,
    policy: str = "stop_loss_first",
    horizons: tuple[int, ...] = (3,),
) -> pd.DataFrame:
    return add_triple_barrier_targets(
        prices,
        atr_length=1,
        stop_atr_multiple=1.0,
        reward_risk_ratio=1.0,
        horizons=horizons,
        entry_price_rule="next_open",
        intrabar_policy=policy,
    )


def test_take_profit_uses_next_open_and_observed_session_horizon() -> None:
    prices = _prices(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 103.0, 99.0, 101.0),
            (101.0, 102.0, 100.0, 101.0),
        ]
    )

    result = _add_targets(prices)

    assert result.iloc[0]["triple_barrier_label_3d"] == 1
    assert result.iloc[0]["time_to_event_3d"] == 2
    assert result.iloc[0]["target_end_date_3d"] == prices.index.get_level_values("trading_date")[2]


def test_entry_uses_next_open_instead_of_signal_close() -> None:
    prices = _prices(
        [
            (110.0, 111.0, 107.0, 109.0),
            (109.0, 110.0, 108.0, 109.0),
            (109.0, 110.0, 108.0, 109.0),
        ]
    )

    result = _add_targets(prices)

    assert result.iloc[0]["triple_barrier_label_3d"] == -1
    assert result.iloc[0]["time_to_event_3d"] == 1


@pytest.mark.parametrize(
    ("gap_open", "expected_label"),
    [(97.0, -1), (103.0, 1)],
)
def test_opening_gap_is_evaluated_before_intrabar_range(
    gap_open: float,
    expected_label: int,
) -> None:
    prices = _prices(
        [
            (100.0, 101.0, 99.0, 100.0),
            (gap_open, 103.0, 96.0, 102.0),
            (100.0, 101.0, 99.0, 100.0),
        ]
    )

    result = _add_targets(prices)

    assert result.iloc[0]["triple_barrier_label_3d"] == expected_label
    assert result.iloc[0]["time_to_event_3d"] == 2


def test_timeout_has_zero_label_and_full_horizon_time() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3)

    result = _add_targets(prices)

    assert result.iloc[0]["triple_barrier_label_3d"] == 0
    assert result.iloc[0]["time_to_event_3d"] == 3
    assert result.iloc[0]["target_end_date_3d"] == prices.index.get_level_values("trading_date")[3]


@pytest.mark.parametrize(
    ("policy", "expected_label"),
    [
        ("stop_loss_first", -1),
        ("take_profit_first", 1),
        ("exclude", pd.NA),
    ],
)
def test_same_bar_policies_are_deterministic(
    policy: str,
    expected_label: object,
) -> None:
    prices = _prices(
        [
            (100.0, 103.0, 97.0, 101.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
        ]
    )

    result = _add_targets(prices, policy=policy)

    if expected_label is pd.NA:
        assert pd.isna(result.iloc[0]["triple_barrier_label_3d"])
        assert pd.isna(result.iloc[0]["time_to_event_3d"])
        assert pd.isna(result.iloc[0]["target_end_date_3d"])
    else:
        assert result.iloc[0]["triple_barrier_label_3d"] == expected_label
        assert result.iloc[0]["time_to_event_3d"] == 1


@pytest.mark.parametrize(
    ("close", "expected_label"),
    [(101.0, -1), (99.0, 1), (100.0, -1)],
)
def test_candle_path_policy_handles_green_red_and_doji(
    close: float,
    expected_label: str,
) -> None:
    prices = _prices(
        [
            (100.0, 103.0, 97.0, close),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
        ]
    )

    result = _add_targets(prices, policy="candle_path")

    assert result.iloc[0]["triple_barrier_label_3d"] == expected_label
    assert result.iloc[0]["time_to_event_3d"] == 1


def test_terminal_rows_remain_nullable() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3)

    result = _add_targets(prices)

    assert result.iloc[1:]["triple_barrier_label_3d"].isna().all()
    assert result.iloc[1:]["time_to_event_3d"].isna().all()
    assert result["triple_barrier_label_3d"].dtype == "Int8"
    assert result["time_to_event_3d"].dtype == "Int64"


def test_terminal_event_is_labeled_when_it_resolves_before_data_ends() -> None:
    prices = _prices([(100.0, 103.0, 99.0, 101.0)])

    result = _add_targets(prices, horizons=(5,))

    assert result.iloc[0]["triple_barrier_label_5d"] == 1
    assert result.iloc[0]["time_to_event_5d"] == 1
    assert result.iloc[0]["target_end_date_5d"] == prices.index.get_level_values("trading_date")[1]


def test_unresolved_terminal_path_remains_unlabeled() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)])

    result = _add_targets(prices, horizons=(5,))

    assert pd.isna(result.iloc[0]["triple_barrier_label_5d"])
    assert pd.isna(result.iloc[0]["time_to_event_5d"])
    assert pd.isna(result.iloc[0]["target_end_date_5d"])


def test_label_and_time_have_matching_missingness() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 4)

    result = _add_targets(prices)

    assert result["triple_barrier_label_3d"].isna().equals(
        result["time_to_event_3d"].isna()
    )


def test_output_schema_contains_one_label_one_time_and_purging_metadata() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3)

    result = _add_targets(prices)

    added_columns = tuple(column for column in result.columns if column not in prices.columns)
    assert added_columns == (
        "triple_barrier_label_3d",
        "time_to_event_3d",
        "target_end_date_3d",
    )


def test_tickers_are_independent_and_canonical_index_is_preserved() -> None:
    take_profit = _prices(
        [(100.0, 103.0, 99.0, 101.0)] + [(100.0, 101.0, 99.0, 100.0)] * 2,
        ticker="AAA",
    )
    stop_loss = _prices(
        [(100.0, 101.0, 97.0, 99.0)] + [(100.0, 101.0, 99.0, 100.0)] * 2,
        ticker="BBB",
    )
    prices = pd.concat([take_profit, stop_loss]).sort_index()

    result = _add_targets(prices)

    assert result.index.equals(prices.index)
    assert (
        result.loc[("test", "AAA", pd.Timestamp("2026-01-02")), "triple_barrier_label_3d"] == 1
    )
    assert (
        result.loc[("test", "BBB", pd.Timestamp("2026-01-02")), "triple_barrier_label_3d"] == -1
    )


def test_invalid_future_ohlc_leaves_label_missing() -> None:
    prices = _prices(
        [
            (100.0, 99.0, 101.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
        ]
    )

    result = _add_targets(prices)

    assert pd.isna(result.iloc[0]["triple_barrier_label_3d"])
    assert pd.isna(result.iloc[0]["time_to_event_3d"])
    assert pd.isna(result.iloc[0]["target_end_date_3d"])


def test_v3_manifest_contains_material_triple_barrier_parameters() -> None:
    barrier_family = V3_TARGET_SET.families[-1]
    manifest = barrier_family.to_manifest()

    assert V3_TARGET_SET.identifier == "ohlcv_price_targets:3"
    assert V3_TARGET_SET.maximum_horizon_sessions == 15
    assert manifest["parameters"] == {
        "atr_length": 14,
        "stop_atr_multiple": 2.0,
        "reward_risk_ratio": 2.0,
        "horizons": [5, 10, 15],
        "intrabar_policy": "stop_loss_first",
    }
    assert barrier_family.output_columns == (
        "triple_barrier_label_5d",
        "time_to_event_5d",
        "target_end_date_5d",
        "triple_barrier_label_10d",
        "time_to_event_10d",
        "target_end_date_10d",
        "triple_barrier_label_15d",
        "time_to_event_15d",
        "target_end_date_15d",
    )
    V3_PRIMARY_TASK.validate_target_set(V3_TARGET_SET)

    changed_values = {
        "atr_length": 10,
        "stop_atr_multiple": 2.5,
        "reward_risk_ratio": 3.0,
        "horizons": (5, 10),
        "intrabar_policy": "take_profit_first",
    }
    for parameter, changed_value in changed_values.items():
        changed_family = TargetFamilySpec(
            name=barrier_family.name,
            builder=barrier_family.builder,
            parameters={**barrier_family.parameters, parameter: changed_value},
            required_columns=barrier_family.required_columns,
            output_columns=barrier_family.output_columns,
            maximum_horizon_sessions=barrier_family.maximum_horizon_sessions,
        )
        changed_set = TargetSetSpec(
            name=V3_TARGET_SET.name,
            version=V3_TARGET_SET.version,
            families=(*V3_TARGET_SET.families[:-1], changed_family),
        )
        assert changed_set.digest != V3_TARGET_SET.digest


def test_generate_v3_labels_executes_all_declared_families() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 29)

    result = generate_v3_labels(prices)

    assert set(V3_TARGET_SET.target_columns).issubset(result.columns)
    assert result.iloc[13]["triple_barrier_label_5d"] == 0
    assert result.iloc[13]["time_to_event_5d"] == 5


def test_parameter_validation_rejects_unsupported_policies() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3)

    with pytest.raises(ValueError, match="Unsupported intrabar_policy"):
        _add_targets(prices, policy="unknown")


def test_invalid_bar_after_resolved_event_does_not_remove_label() -> None:
    prices = _prices(
        [
            (100.0, 103.0, 99.0, 101.0),
            (100.0, 99.0, 101.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
        ]
    )

    result = _add_targets(prices)

    assert result.iloc[0]["triple_barrier_label_3d"] == 1
    assert result.iloc[0]["time_to_event_3d"] == 1


def test_adjustment_consistent_prices_make_split_encoding_invariant() -> None:
    baseline = _prices(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 103.0, 99.0, 101.0),
            (101.0, 102.0, 100.0, 101.0),
        ]
    )
    split_encoded = baseline.copy()
    ohlc_positions = split_encoded.columns.get_indexer(["open", "high", "low", "close"])
    split_encoded.iloc[0, ohlc_positions] *= 2.0
    split_encoded.iloc[0, split_encoded.columns.get_loc("adjusted_close")] = 100.0

    baseline_result = _add_targets(baseline)
    split_result = _add_targets(split_encoded)

    columns = [
        "triple_barrier_label_3d",
        "time_to_event_3d",
        "target_end_date_3d",
    ]
    pd.testing.assert_series_equal(
        baseline_result.iloc[0][columns],
        split_result.iloc[0][columns],
        check_names=False,
    )


def test_builder_does_not_mutate_input_and_preserves_canonical_index() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3)
    original = prices.copy(deep=True)

    result = _add_targets(prices)

    pd.testing.assert_frame_equal(prices, original)
    assert result.index.equals(prices.index)


def test_empty_input_has_stable_nullable_output_dtypes() -> None:
    empty = _prices([]).iloc[0:0]

    result = _add_targets(empty)

    assert result["triple_barrier_label_3d"].dtype == "Int8"
    assert result["time_to_event_3d"].dtype == "Int64"
    assert result["target_end_date_3d"].dtype == "datetime64[ns]"


def test_duplicate_observation_keys_are_rejected() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3)
    duplicate = pd.concat([prices.iloc[[0]], prices])

    with pytest.raises(ValueError, match="must have a unique index"):
        _add_targets(duplicate)


def test_unsorted_canonical_index_is_rejected() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3).iloc[::-1]

    with pytest.raises(ValueError, match="must be sorted"):
        _add_targets(prices)


def test_flat_identifier_columns_are_rejected() -> None:
    prices = _prices([(100.0, 101.0, 99.0, 100.0)] * 3).reset_index()

    with pytest.raises(ValueError, match="must use a MultiIndex"):
        _add_targets(prices)
