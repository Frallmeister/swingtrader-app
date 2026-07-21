import pandas as pd
import pytest

from swingtrader.data.features.market_structure import add_market_structure_features
from swingtrader.data.market_frame import validate_new_columns


def test_validate_new_columns_rejects_existing_columns_in_sorted_order() -> None:
    data = pd.DataFrame({"feature_b": [1.0], "feature_a": [2.0]})

    with pytest.raises(
        ValueError,
        match=("Generated columns already exist in input data: feature_a, feature_b\\."),
    ):
        validate_new_columns(
            data,
            new_columns=("feature_b", "missing", "feature_a"),
        )


def test_add_market_structure_features_rejects_existing_feature_column() -> None:
    index = pd.MultiIndex.from_product(
        [["yfinance"], ["AAA.ST"], pd.date_range("2026-01-01", periods=5)],
        names=["provider", "ticker", "trading_date"],
    )
    data = pd.DataFrame(
        {
            "high": [10.0, 11.0, 10.0, 12.0, 11.0],
            "low": [9.0, 10.0, 9.5, 10.5, 10.0],
            "close": [9.5, 10.5, 9.8, 11.5, 10.5],
            "zigzag_retracement": [0.0] * 5,
        },
        index=index,
    )

    with pytest.raises(
        ValueError,
        match="Generated columns already exist in input data: zigzag_retracement\\.",
    ):
        add_market_structure_features(data, zigzag_pivot_legs=2)
