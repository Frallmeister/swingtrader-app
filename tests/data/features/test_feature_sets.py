import json

import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features import (
    DEFAULT_FEATURE_SET,
    FeatureBlockSpec,
    FeatureSetSpec,
    HistoryRequirement,
    add_default_features,
    add_feature_set,
)


def test_default_feature_set_matches_pipeline_output_schema() -> None:
    prices = _prices()

    result = add_feature_set(prices)

    appended_columns = tuple(result.columns[len(prices.columns) :])
    assert appended_columns == DEFAULT_FEATURE_SET.feature_columns
    assert result.columns.is_unique


def test_add_default_features_uses_the_versioned_default_set() -> None:
    prices = _prices()

    expected = add_feature_set(prices, feature_set=DEFAULT_FEATURE_SET)
    result = add_default_features(prices)

    pd.testing.assert_frame_equal(result, expected)


def test_selected_feature_set_runs_only_requested_blocks() -> None:
    prices = _prices()
    selected = DEFAULT_FEATURE_SET.select(
        "returns",
        "volume",
        name="returns_and_volume",
        version="1",
    )

    result = add_feature_set(prices, feature_set=selected)

    assert selected.identifier == "returns_and_volume:1"
    assert tuple(result.columns[len(prices.columns) :]) == (
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
        "turnover_zscore",
    )


def test_feature_set_manifest_is_deterministic_and_json_serializable() -> None:
    manifest = DEFAULT_FEATURE_SET.to_manifest()

    assert manifest == DEFAULT_FEATURE_SET.to_manifest()
    json.dumps(manifest)

    assert manifest["name"] == "ohlcv_v1_candidates"
    assert manifest["version"] == "1"

    history_requirements = {
        block["name"]: block["history_requirement"] for block in manifest["blocks"]
    }

    assert history_requirements == {
        "returns": "bounded",
        "trend": "expanding",
        "momentum": "expanding",
        "volatility": "expanding",
        "price_action": "expanding",
        "volume": "bounded",
        "market_structure": "path_dependent",
    }


def test_feature_set_rejects_duplicate_output_columns() -> None:
    first = _block("first", output_columns=("duplicate",))
    second = _block("second", output_columns=("duplicate",))

    with pytest.raises(ValueError, match="unique across a feature set"):
        FeatureSetSpec(name="invalid", version="1", blocks=(first, second))


def test_feature_set_rejects_unknown_selected_blocks() -> None:
    with pytest.raises(ValueError, match="Unknown feature block names: missing"):
        DEFAULT_FEATURE_SET.select(
            "missing",
            name="invalid",
            version="1",
        )


def test_feature_block_copies_output_columns() -> None:
    output_columns = ["feature"]

    def builder(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    block = FeatureBlockSpec(
        name="block",
        builder=builder,
        output_columns=output_columns,  # type: ignore[arg-type]
    )

    output_columns.append("later_mutation")

    assert block.output_columns == ("feature",)


def test_feature_set_copies_blocks() -> None:
    first = _block("first", output_columns=("first_feature",))
    second = _block("second", output_columns=("second_feature",))
    blocks = [first]

    feature_set = FeatureSetSpec(
        name="stable",
        version="1",
        blocks=blocks,  # type: ignore[arg-type]
    )

    blocks.append(second)

    assert feature_set.blocks == (first,)


def _block(
    name: str,
    *,
    output_columns: tuple[str, ...],
) -> FeatureBlockSpec:
    def builder(data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()

    return FeatureBlockSpec(
        name=name,
        builder=builder,
        output_columns=output_columns,
        history_requirement=HistoryRequirement.BOUNDED,
    )


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    observation_count = 140
    trading_dates = [
        timestamp.date()
        for timestamp in pd.date_range(
            "2026-01-01",
            periods=observation_count,
            freq="B",
        )
    ]
    frames = []

    for ticker, base in (("AAA.ST", 100.0), ("BBB.ST", 50.0)):
        steps = rng.normal(0.0, 1.0, observation_count)
        close = base + np.cumsum(steps)
        span = np.abs(rng.normal(0.0, 1.0, observation_count)) + 0.5
        frame = pd.DataFrame(
            {
                "open": close + rng.uniform(-0.5, 0.5, observation_count) * span,
                "high": close + span,
                "low": close - span,
                "close": close,
                "adjusted_close": close,
                "volume": rng.integers(
                    1_000,
                    5_000,
                    observation_count,
                ).astype(float),
            }
        )
        frame.index = pd.MultiIndex.from_arrays(
            [
                ["yfinance"] * observation_count,
                [ticker] * observation_count,
                trading_dates,
            ],
            names=["provider", "ticker", "trading_date"],
        )
        frames.append(frame)

    return pd.concat(frames).sort_index()
