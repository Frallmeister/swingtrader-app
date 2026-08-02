import json

import pandas as pd
import pytest

from swingtrader.modeling.datasets import (
    CROSS_SECTIONAL_RETURN_PRIMARY_TASK,
    CROSS_SECTIONAL_RETURN_TARGET_SET,
    FORWARD_RETURN_PRIMARY_TASK,
    FORWARD_RETURN_TARGET_SET,
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
    generate_cross_sectional_return_labels,
    generate_forward_return_labels,
    generate_target_set,
)


def test_forward_return_target_set_manifest_is_deterministic_and_serializable() -> None:
    manifest = FORWARD_RETURN_TARGET_SET.to_manifest()
    assert manifest == FORWARD_RETURN_TARGET_SET.to_manifest()
    json.dumps(manifest)
    assert FORWARD_RETURN_TARGET_SET.identifier == "forward_return_targets:1"
    assert FORWARD_RETURN_TARGET_SET.family_names == ("forward_returns", "significant_up_5d")
    assert FORWARD_RETURN_TARGET_SET.maximum_horizon_sessions == 15


def test_cross_sectional_return_target_set_contains_only_cross_sectional_family() -> None:
    assert CROSS_SECTIONAL_RETURN_TARGET_SET.identifier == "cross_sectional_return_targets:1"
    assert CROSS_SECTIONAL_RETURN_TARGET_SET.family_names == ("cross_sectional_returns",)
    assert CROSS_SECTIONAL_RETURN_TARGET_SET.maximum_horizon_sessions == 15

    target_columns = CROSS_SECTIONAL_RETURN_TARGET_SET.target_columns
    assert "market_relative_forward_return_5d" in target_columns
    assert "forward_return_5d_relevance_grade" in target_columns
    assert "forward_return_5d_cross_sectional_percentile" in target_columns
    assert "target_significant_up_5d" not in target_columns
    assert "triple_barrier_label_5d" not in target_columns

    CROSS_SECTIONAL_RETURN_PRIMARY_TASK.validate_target_set(CROSS_SECTIONAL_RETURN_TARGET_SET)
    assert (
        CROSS_SECTIONAL_RETURN_PRIMARY_TASK.target_column
        == "forward_return_5d_cross_sectional_percentile"
    )
    assert CROSS_SECTIONAL_RETURN_PRIMARY_TASK.task_type == "regression"


def test_meaningful_parameter_change_changes_digest() -> None:
    family = FORWARD_RETURN_TARGET_SET.families[0]
    changed = TargetFamilySpec(
        name=family.name,
        builder=family.builder,
        parameters={"horizons": (5, 10)},
        required_columns=family.required_columns,
        output_columns=("forward_return_5d", "forward_return_10d"),
        maximum_horizon_sessions=10,
    )
    target_set = TargetSetSpec(
        name=FORWARD_RETURN_TARGET_SET.name, version="2", families=(changed,)
    )
    assert target_set.digest != FORWARD_RETURN_TARGET_SET.digest


def test_target_set_rejects_duplicate_family_names() -> None:
    family = FORWARD_RETURN_TARGET_SET.families[0]
    with pytest.raises(ValueError, match="family names must be unique"):
        TargetSetSpec(name="invalid", version="1", families=(family, family))


def test_target_set_rejects_output_collisions() -> None:
    first = _family("first", "duplicate")
    second = _family("second", "duplicate")
    with pytest.raises(ValueError, match="output columns must be unique"):
        TargetSetSpec(name="invalid", version="1", families=(first, second))


def test_target_family_rejects_invalid_horizon() -> None:
    with pytest.raises(ValueError, match="at least one session"):
        _family("invalid", "target", maximum_horizon_sessions=0)


def test_target_family_rejects_unknown_parameters() -> None:
    with pytest.raises(ValueError, match="Unknown parameters"):
        TargetFamilySpec(
            name="invalid",
            builder=lambda data: data.copy(),
            parameters={"missing": 1},
            output_columns=("target",),
            maximum_horizon_sessions=1,
        )


def test_target_family_rejects_missing_required_parameters() -> None:
    with pytest.raises(ValueError, match="Missing required parameters"):
        TargetFamilySpec(
            name="invalid",
            builder=add_required_target,
            output_columns=("target",),
            maximum_horizon_sessions=1,
        )


def test_execution_rejects_output_overwrite() -> None:
    prices = pd.DataFrame({"existing": [1]})
    target_set = TargetSetSpec(
        name="invalid",
        version="1",
        families=(_family("overwrite", "existing"),),
    )
    with pytest.raises(ValueError, match="would overwrite columns"):
        generate_target_set(prices, target_set=target_set)


def test_forward_return_task_selects_one_generated_target() -> None:
    FORWARD_RETURN_PRIMARY_TASK.validate_target_set(FORWARD_RETURN_TARGET_SET)
    assert FORWARD_RETURN_PRIMARY_TASK.target_column == "target_significant_up_5d"
    assert FORWARD_RETURN_PRIMARY_TASK.task_type == "classification"


def test_task_rejects_unknown_target_column() -> None:
    task = SupervisedTaskSpec(
        name="invalid",
        target_set_name=FORWARD_RETURN_TARGET_SET.name,
        target_set_version=FORWARD_RETURN_TARGET_SET.version,
        target_column="missing",
        task_type="classification",
    )
    with pytest.raises(ValueError, match="Unknown target column"):
        task.validate_target_set(FORWARD_RETURN_TARGET_SET)


def test_target_families_execute_in_declared_order() -> None:
    target_set = TargetSetSpec(
        name="ordered",
        version="1",
        families=(
            TargetFamilySpec(
                name="first",
                builder=add_first_target,
                output_columns=("first",),
                maximum_horizon_sessions=1,
            ),
            TargetFamilySpec(
                name="second",
                builder=add_second_target,
                required_columns=frozenset({"first"}),
                output_columns=("second",),
                maximum_horizon_sessions=1,
            ),
        ),
    )

    result = generate_target_set(pd.DataFrame(index=[0]), target_set=target_set)

    assert result.loc[0, "first"] == 1
    assert result.loc[0, "second"] == 2


def test_forward_return_wrapper_delegates_to_forward_return_target_set() -> None:
    prices = (
        pd.DataFrame(
            {
                "provider": ["yfinance"] * 16,
                "ticker": ["AAA.ST"] * 16,
                "trading_date": pd.date_range("2026-01-01", periods=16),
                "adjusted_close": range(100, 116),
            }
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        generate_target_set(prices, target_set=FORWARD_RETURN_TARGET_SET),
        generate_forward_return_labels(prices),
    )


def test_cross_sectional_set_executes_independently_from_canonical_prices() -> None:
    frames = []
    for ticker, base in (("AAA.ST", 100), ("BBB.ST", 200)):
        frames.append(
            pd.DataFrame(
                {
                    "provider": ["yfinance"] * 16,
                    "ticker": [ticker] * 16,
                    "trading_date": pd.date_range("2026-01-01", periods=16),
                    "adjusted_close": range(base, base + 16),
                }
            )
        )
    prices = pd.concat(frames).set_index(["provider", "ticker", "trading_date"]).sort_index()

    result = generate_cross_sectional_return_labels(prices)

    assert "forward_return_5d_cross_sectional_percentile" in result.columns
    assert "market_relative_forward_return_5d" in result.columns
    assert "forward_return_5d_relevance_grade" in result.columns
    assert "target_significant_up_5d" not in result.columns
    assert "triple_barrier_label_5d" not in result.columns
    added_columns = set(result.columns).difference(prices.columns)
    assert added_columns == set(CROSS_SECTIONAL_RETURN_TARGET_SET.target_columns)


def add_required_target(data: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    result = data.copy()
    result["target"] = threshold
    return result


def add_first_target(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["first"] = 1
    return result


def add_second_target(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["second"] = result["first"] + 1
    return result


def _family(
    name: str,
    output_column: str,
    *,
    maximum_horizon_sessions: int = 1,
) -> TargetFamilySpec:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    return TargetFamilySpec(
        name=name,
        builder=builder,
        output_columns=(output_column,),
        maximum_horizon_sessions=maximum_horizon_sessions,
    )
