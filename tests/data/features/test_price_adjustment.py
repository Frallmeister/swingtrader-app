from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features._price_adjustment import (
    adjustment_consistent_price_frame,
)
from swingtrader.data.features.market_structure import add_market_structure_features
from swingtrader.data.features.momentum import add_momentum_features
from swingtrader.data.features.price_action import add_price_action_features
from swingtrader.data.features.returns import add_return_features
from swingtrader.data.features.trend import add_trend_features
from swingtrader.data.features.volatility import add_volatility_features
from swingtrader.data.features.volume import add_volume_features


def test_adjustment_consistent_price_frame_preserves_geometry_and_input() -> None:
    prices = pd.DataFrame(
        {
            "open": [200.0, 51.0],
            "high": [210.0, 53.0],
            "low": [190.0, 49.0],
            "close": [204.0, 52.0],
            "adjusted_close": [102.0, 52.0],
        },
        index=pd.Index(["before", "after"]),
    )
    original = prices.copy(deep=True)

    result = adjustment_consistent_price_frame(
        prices,
        price_columns=("open", "high", "low", "close"),
    )

    expected = pd.DataFrame(
        {
            "open": [100.0, 51.0],
            "high": [105.0, 53.0],
            "low": [95.0, 49.0],
            "close": [102.0, 52.0],
        },
        index=prices.index,
    )
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(prices, original)


_FEATURE_FAMILIES: list[tuple[Callable[..., pd.DataFrame], dict[str, Any], str]] = [
    (
        add_return_features,
        {"horizons": (1, 5, 10)},
        "return_1d",
    ),
    (
        add_trend_features,
        {
            "ma_lengths": (3, 5, 8),
            "adx_length": 5,
            "vwap_length": 5,
            "vwap_bollinger_length": 5,
        },
        "adx",
    ),
    (
        add_momentum_features,
        {
            "ppo_lengths": (3, 6, 3),
            "ppo_percentile_min_history": 5,
            "rsi_length": 5,
            "rsi_bollinger_length": 5,
            "stochastic_k_length": 5,
            "stochastic_k_smoothing": 2,
            "stochastic_d_length": 2,
            "mfi_length": 5,
            "mfi_bollinger_length": 5,
            "squeeze_bb_length": 5,
            "squeeze_kc_length": 5,
            "squeeze_atr_length": 5,
        },
        "ppo",
    ),
    (
        add_volatility_features,
        {
            "adr_length": 5,
            "atr_length": 5,
            "bollinger_length": 5,
        },
        "atr_percent",
    ),
    (
        add_price_action_features,
        {
            "atr_length": 5,
            "range_percentile_length": 5,
            "breakout_length": 5,
        },
        "candle_signed_body_fraction",
    ),
    (
        add_volume_features,
        {
            "turnover_zscore_length": 5,
        },
        "turnover_zscore",
    ),
    (
        add_market_structure_features,
        {
            "zigzag_deviation": 1.0,
            "zigzag_pivot_legs": 4,
            "zigzag_consistency_pivots": 3,
            "zigzag_dynamics_legs": 4,
            "zigzag_atr_length": 5,
        },
        "zigzag_last_direction",
    ),
]


@pytest.mark.parametrize(
    ("builder", "kwargs", "collision_column"),
    _FEATURE_FAMILIES,
)
def test_feature_family_is_invariant_to_split_encoded_raw_prices(
    builder: Callable[..., pd.DataFrame],
    kwargs: dict[str, Any],
    collision_column: str,
) -> None:
    del collision_column
    continuous = _continuous_prices()
    split_encoded = continuous.copy(deep=True)
    before_split = split_encoded.index.get_level_values("trading_date") < pd.Timestamp("2025-03-10")
    split_encoded.loc[before_split, ["open", "high", "low", "close"]] *= 2.0

    continuous_result = builder(continuous, **kwargs)
    split_result = builder(split_encoded, **kwargs)
    feature_columns = [
        column for column in continuous_result.columns if column not in continuous.columns
    ]

    assert feature_columns
    assert continuous_result.loc[:, feature_columns].notna().any().any()
    pd.testing.assert_frame_equal(
        split_result.loc[:, feature_columns],
        continuous_result.loc[:, feature_columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("builder", "kwargs", "collision_column"),
    _FEATURE_FAMILIES,
)
def test_feature_family_rejects_existing_generated_columns(
    builder: Callable[..., pd.DataFrame],
    kwargs: dict[str, Any],
    collision_column: str,
) -> None:
    prices = _continuous_prices()
    prices[collision_column] = 0.0

    with pytest.raises(
        ValueError,
        match=f"Generated columns already exist in input data: {collision_column}",
    ):
        builder(prices, **kwargs)


@pytest.mark.parametrize(
    ("builder", "kwargs", "collision_column"),
    _FEATURE_FAMILIES,
)
def test_future_rows_do_not_change_existing_feature_values(
    builder: Callable[..., pd.DataFrame],
    kwargs: dict[str, Any],
    collision_column: str,
) -> None:
    del collision_column
    prices = _continuous_prices()
    prefix = prices.iloc[:80]

    full_result = builder(prices, **kwargs)
    prefix_result = builder(prefix, **kwargs)
    feature_columns = [column for column in full_result.columns if column not in prices.columns]

    pd.testing.assert_frame_equal(
        full_result.loc[prefix.index, feature_columns],
        prefix_result.loc[:, feature_columns],
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def _continuous_prices() -> pd.DataFrame:
    observations = 120
    position = np.arange(observations, dtype="float64")
    close = 100.0 + 0.08 * position + 8.0 * np.sin(position * np.pi / 10.0)
    open_ = close * (1.0 + 0.004 * np.cos(position * np.pi / 4.0))
    spread = 1.0 + 0.25 * np.sin(position * np.pi / 7.0) ** 2
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1_000_000.0 + 100_000.0 * (1.0 + np.sin(position * np.pi / 6.0))
    dates = pd.date_range("2025-01-01", periods=observations, freq="D")

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": close,
            "volume": volume,
        },
        index=pd.MultiIndex.from_arrays(
            [
                np.repeat("yfinance", observations),
                np.repeat("AAA.ST", observations),
                dates,
            ],
            names=["provider", "ticker", "trading_date"],
        ),
    )
