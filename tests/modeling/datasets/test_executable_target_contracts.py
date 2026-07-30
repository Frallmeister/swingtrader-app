import pandas as pd
import pytest

from swingtrader.modeling.datasets.contracts import TargetFamilySpec, TargetSetSpec


def test_target_family_passes_parameters_and_selects_declared_outputs() -> None:
    def builder(data: pd.DataFrame, *, horizon: int) -> pd.DataFrame:
        result = data.copy()
        result["diagnostic"] = -1
        result["target_end_date"] = pd.Timestamp("2026-01-01")
        result["target"] = result["value"].shift(-horizon)
        return result

    family = TargetFamilySpec(
        name="future_value",
        builder=builder,
        parameters={"horizon": 1},
        required_columns=frozenset({"value"}),
        output_columns=("target", "target_end_date"),
        maximum_horizon_sessions=1,
    )

    result = family.apply(pd.DataFrame({"value": [1, 2]}))

    assert list(result.columns) == ["target", "target_end_date"]
    assert result["target"].iloc[0] == 2
    assert "diagnostic" not in result


def test_target_family_requires_explicit_optional_parameters() -> None:
    def builder(data: pd.DataFrame, *, horizon: int = 2) -> pd.DataFrame:
        result = data.copy()
        result["target"] = horizon
        return result

    with pytest.raises(ValueError, match="Missing required parameters for target family"):
        TargetFamilySpec(
            name="defaults",
            builder=builder,
            output_columns=("target",),
            maximum_horizon_sessions=2,
        )


def test_target_set_executes_families_in_order_and_preserves_input() -> None:
    def add_first(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["first"] = result["value"].shift(-1)
        return result

    def add_second(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["second"] = result["first"].gt(result["value"])
        return result

    target_set = TargetSetSpec(
        name="ordered",
        version="1",
        families=(
            TargetFamilySpec(
                name="first",
                builder=add_first,
                required_columns=frozenset({"value"}),
                output_columns=("first",),
                maximum_horizon_sessions=1,
            ),
            TargetFamilySpec(
                name="second",
                builder=add_second,
                required_columns=frozenset({"first", "value"}),
                output_columns=("second",),
                maximum_horizon_sessions=1,
            ),
        ),
    )
    data = pd.DataFrame({"value": [1, 2]})
    original = data.copy(deep=True)

    result = target_set.apply(data)

    assert target_set.source_columns == ("value",)
    assert list(result.columns) == ["value", "first", "second"]
    assert result["second"].iloc[0]
    pd.testing.assert_frame_equal(data, original)


def test_target_family_accepts_required_index_level() -> None:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["target"] = 1
        return result

    family = TargetFamilySpec(
        name="index_input",
        builder=builder,
        required_columns=frozenset({"ticker"}),
        output_columns=("target",),
        maximum_horizon_sessions=1,
    )
    data = pd.DataFrame(
        {"value": [1]},
        index=pd.Index(["AAA"], name="ticker"),
    )

    assert family.apply(data)["target"].tolist() == [1]


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (lambda data: data.copy(), "did not produce columns"),
        (lambda data: data.iloc[::-1], "changed the canonical sample index"),
        (lambda data: pd.Series([1]), "must return a pandas DataFrame"),
    ],
)
def test_target_family_rejects_invalid_builder_results(
    builder: object,
    match: str,
) -> None:
    family = TargetFamilySpec(
        name="invalid_result",
        builder=builder,  # type: ignore[arg-type]
        output_columns=("target",),
        maximum_horizon_sessions=1,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        family.apply(pd.DataFrame({"value": [1, 2]}))


def test_target_family_rejects_duplicate_outputs_and_changed_index_names() -> None:
    def duplicate_outputs(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([[1, 2]], index=data.index, columns=["target", "target"])

    duplicate_family = TargetFamilySpec(
        name="duplicates",
        builder=duplicate_outputs,
        output_columns=("target",),
        maximum_horizon_sessions=1,
    )
    with pytest.raises(ValueError, match="returned duplicate columns"):
        duplicate_family.apply(pd.DataFrame({"value": [1]}))

    def rename_index(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result.index = result.index.rename("changed")
        result["target"] = 1
        return result

    renamed_family = TargetFamilySpec(
        name="renamed_index",
        builder=rename_index,
        output_columns=("target",),
        maximum_horizon_sessions=1,
    )
    source = pd.DataFrame({"value": [1]}, index=pd.Index([0], name="sample"))
    with pytest.raises(ValueError, match="changed the canonical sample index"):
        renamed_family.apply(source)


def test_target_family_rejects_missing_inputs_and_output_collisions() -> None:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["target"] = 1
        return result

    missing_input = TargetFamilySpec(
        name="missing_input",
        builder=builder,
        required_columns=frozenset({"required"}),
        output_columns=("target",),
        maximum_horizon_sessions=1,
    )
    with pytest.raises(ValueError, match="missing required inputs: required"):
        missing_input.apply(pd.DataFrame({"value": [1]}))

    collision = TargetFamilySpec(
        name="collision",
        builder=builder,
        output_columns=("target",),
        maximum_horizon_sessions=1,
    )
    with pytest.raises(ValueError, match="would overwrite columns: target"):
        collision.apply(pd.DataFrame({"target": [1]}))
