import pandas as pd
import pytest

from swingtrader.data.features.contracts import FeatureBlockSpec, FeatureSetSpec


def test_feature_block_passes_parameters_and_selects_declared_outputs() -> None:
    def builder(data: pd.DataFrame, *, multiplier: int) -> pd.DataFrame:
        result = data.copy()
        result["undeclared"] = -1
        result["second"] = result["value"] * multiplier
        result["first"] = result["value"] + multiplier
        return result

    block = FeatureBlockSpec(
        name="calculation",
        builder=builder,
        parameters={"multiplier": 3},
        required_columns=frozenset({"value"}),
        output_columns=("first", "second"),
    )

    result = block.apply(pd.DataFrame({"value": [1, 2]}))

    assert list(result.columns) == ["first", "second"]
    assert result.to_dict("list") == {"first": [4, 5], "second": [3, 6]}


def test_feature_block_requires_explicit_optional_parameters() -> None:
    def builder(data: pd.DataFrame, *, multiplier: int = 4) -> pd.DataFrame:
        result = data.copy()
        result["feature"] = result["value"] * multiplier
        return result

    with pytest.raises(ValueError, match="Missing required parameters for feature block"):
        FeatureBlockSpec(
            name="defaults",
            builder=builder,
            output_columns=("feature",),
        )


def test_feature_set_executes_blocks_in_order_and_preserves_input() -> None:
    def add_first(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["first"] = result["value"] + 1
        return result

    def add_second(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["second"] = result["first"] * 2
        return result

    feature_set = FeatureSetSpec(
        name="ordered",
        version="1",
        blocks=(
            FeatureBlockSpec(
                name="first",
                builder=add_first,
                required_columns=frozenset({"value"}),
                output_columns=("first",),
            ),
            FeatureBlockSpec(
                name="second",
                builder=add_second,
                required_columns=frozenset({"first"}),
                output_columns=("second",),
            ),
        ),
    )
    data = pd.DataFrame({"value": [2, 4]})
    original = data.copy(deep=True)

    result = feature_set.apply(data)

    assert feature_set.source_columns == ("value",)
    assert list(result.columns) == ["value", "first", "second"]
    assert result["second"].tolist() == [6, 10]
    pd.testing.assert_frame_equal(data, original)


def test_feature_block_isolates_builder_mutation_from_the_source() -> None:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        data.iloc[0, data.columns.get_loc("value")] = -1
        data["feature"] = data["value"]
        return data

    block = FeatureBlockSpec(
        name="isolated",
        builder=builder,
        required_columns=frozenset({"value"}),
        output_columns=("feature",),
    )
    data = pd.DataFrame({"value": [1, 2]})
    original = data.copy(deep=True)

    result = block.apply(data)

    pd.testing.assert_frame_equal(data, original)
    assert result["feature"].tolist() == [-1, 2]


def test_feature_block_accepts_required_index_level() -> None:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["feature"] = 1
        return result

    block = FeatureBlockSpec(
        name="index_input",
        builder=builder,
        required_columns=frozenset({"ticker"}),
        output_columns=("feature",),
    )
    data = pd.DataFrame(
        {"value": [1]},
        index=pd.Index(["AAA"], name="ticker"),
    )

    assert block.apply(data)["feature"].tolist() == [1]


def test_feature_block_rejects_invalid_parameters() -> None:
    def builder(data: pd.DataFrame, *, required: int) -> pd.DataFrame:
        return data.copy()

    with pytest.raises(ValueError, match="Unknown parameters for feature block"):
        FeatureBlockSpec(
            name="unknown",
            builder=builder,
            parameters={"other": 1},
            output_columns=("feature",),
        )

    with pytest.raises(ValueError, match="Missing required parameters for feature block"):
        FeatureBlockSpec(
            name="missing",
            builder=builder,
            output_columns=("feature",),
        )


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (lambda data: data.copy(), "did not produce columns"),
        (lambda data: data.iloc[::-1], "changed the canonical sample index"),
        (lambda data: pd.Series([1]), "must return a pandas DataFrame"),
    ],
)
def test_feature_block_rejects_invalid_builder_results(
    builder: object,
    match: str,
) -> None:
    block = FeatureBlockSpec(
        name="invalid_result",
        builder=builder,  # type: ignore[arg-type]
        output_columns=("feature",),
    )

    with pytest.raises((TypeError, ValueError), match=match):
        block.apply(pd.DataFrame({"value": [1, 2]}))


def test_feature_block_rejects_invalid_input_frame() -> None:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["feature"] = 1
        return result

    block = FeatureBlockSpec(
        name="input",
        builder=builder,
        output_columns=("feature",),
    )

    with pytest.raises(TypeError, match="input must be a pandas DataFrame"):
        block.apply(pd.Series([1]))  # type: ignore[arg-type]

    duplicate_columns = pd.DataFrame([[1, 2]], columns=["value", "value"])
    with pytest.raises(ValueError, match="input contains duplicate columns"):
        block.apply(duplicate_columns)


def test_feature_block_rejects_duplicate_outputs_and_changed_index_names() -> None:
    def duplicate_outputs(data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([[1, 2]], index=data.index, columns=["feature", "feature"])

    duplicate_block = FeatureBlockSpec(
        name="duplicates",
        builder=duplicate_outputs,
        output_columns=("feature",),
    )
    with pytest.raises(ValueError, match="returned duplicate columns"):
        duplicate_block.apply(pd.DataFrame({"value": [1]}))

    def rename_index(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result.index = result.index.rename("changed")
        result["feature"] = 1
        return result

    renamed_block = FeatureBlockSpec(
        name="renamed_index",
        builder=rename_index,
        output_columns=("feature",),
    )
    source = pd.DataFrame({"value": [1]}, index=pd.Index([0], name="sample"))
    with pytest.raises(ValueError, match="changed the canonical sample index"):
        renamed_block.apply(source)

    def change_index_dtype(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result.index = result.index.astype("uint64")
        result["feature"] = 1
        return result

    dtype_block = FeatureBlockSpec(
        name="changed_index_dtype",
        builder=change_index_dtype,
        output_columns=("feature",),
    )
    with pytest.raises(ValueError, match="changed the canonical sample index"):
        dtype_block.apply(pd.DataFrame({"value": [1]}, index=pd.Index([1], dtype="int64")))


def test_feature_block_rejects_missing_inputs_and_output_collisions() -> None:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["feature"] = 1
        return result

    missing_input = FeatureBlockSpec(
        name="missing_input",
        builder=builder,
        required_columns=frozenset({"required"}),
        output_columns=("feature",),
    )
    with pytest.raises(ValueError, match="missing required inputs: required"):
        missing_input.apply(pd.DataFrame({"value": [1]}))

    collision = FeatureBlockSpec(
        name="collision",
        builder=builder,
        output_columns=("feature",),
    )
    with pytest.raises(ValueError, match="would overwrite columns: feature"):
        collision.apply(pd.DataFrame({"feature": [1]}))
