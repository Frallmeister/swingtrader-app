import json

import pandas as pd
import pytest

from swingtrader.modeling.datasets import (
    V1_PRIMARY_TASK,
    V1_TARGET_SET,
    SupervisedTaskSpec,
    TargetFamilySpec,
    TargetSetSpec,
    generate_target_set,
    generate_v1_labels,
)


def test_v1_target_set_manifest_is_deterministic_and_serializable() -> None:
    manifest = V1_TARGET_SET.to_manifest()
    assert manifest == V1_TARGET_SET.to_manifest()
    json.dumps(manifest)
    assert V1_TARGET_SET.identifier == "ohlcv_price_targets:1"
    assert V1_TARGET_SET.family_names == ("forward_returns", "significant_up_5d")
    assert V1_TARGET_SET.maximum_horizon_sessions == 15


def test_meaningful_parameter_change_changes_digest() -> None:
    family = V1_TARGET_SET.families[0]
    changed = TargetFamilySpec(
        name=family.name,
        builder=family.builder,
        parameters={"horizons": (5, 10)},
        required_columns=family.required_columns,
        output_columns=("forward_return_5d", "forward_return_10d"),
        maximum_horizon_sessions=10,
    )
    target_set = TargetSetSpec(name=V1_TARGET_SET.name, version="2", families=(changed,))
    assert target_set.digest != V1_TARGET_SET.digest


def test_target_set_rejects_duplicate_family_names() -> None:
    family = V1_TARGET_SET.families[0]
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


def test_execution_rejects_output_overwrite() -> None:
    prices = pd.DataFrame({"existing": [1]})
    target_set = TargetSetSpec(
        name="invalid",
        version="1",
        families=(_family("overwrite", "existing"),),
    )
    with pytest.raises(ValueError, match="would overwrite columns"):
        generate_target_set(prices, target_set=target_set)


def test_v1_task_selects_one_generated_target() -> None:
    V1_PRIMARY_TASK.validate_target_set(V1_TARGET_SET)
    assert V1_PRIMARY_TASK.target_column == "target_significant_up_5d"
    assert V1_PRIMARY_TASK.task_type == "classification"


def test_task_rejects_unknown_target_column() -> None:
    task = SupervisedTaskSpec(
        name="invalid",
        target_set_name=V1_TARGET_SET.name,
        target_set_version=V1_TARGET_SET.version,
        target_column="missing",
        task_type="classification",
    )
    with pytest.raises(ValueError, match="Unknown target column"):
        task.validate_target_set(V1_TARGET_SET)


def test_target_set_execution_matches_v1_compatibility_wrapper() -> None:
    prices = pd.DataFrame(
        {
            "provider": ["yfinance"] * 16,
            "ticker": ["AAA.ST"] * 16,
            "trading_date": pd.date_range("2026-01-01", periods=16),
            "adjusted_close": range(100, 116),
        }
    )
    pd.testing.assert_frame_equal(
        generate_target_set(prices, target_set=V1_TARGET_SET),
        generate_v1_labels(prices),
    )


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
