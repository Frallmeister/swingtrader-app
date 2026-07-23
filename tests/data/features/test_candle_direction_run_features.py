import pandas as pd
import pytest

from swingtrader.data.features.price_action import add_price_action_features
from swingtrader.indicators import candle_direction_runs


def test_add_price_action_features_adds_adjusted_direction_run_outputs() -> None:
    data = _prices()

    result = add_price_action_features(
        data,
        atr_length=1,
        range_percentile_length=1,
        breakout_length=1,
    )
    adjustment_factor = data["adjusted_close"] / data["close"]
    adjusted_ohlc = data[["open", "high", "low", "close"]].mul(
        adjustment_factor,
        axis=0,
    )
    expected = candle_direction_runs(adjusted_ohlc, atr_length=1)

    feature_names = {
        "direction_run": "candle_direction_run",
        "direction_run_return": "candle_direction_run_return",
        "direction_run_body_atr": "candle_direction_run_body_atr",
    }
    for indicator_name, feature_name in feature_names.items():
        pd.testing.assert_series_equal(
            result[feature_name],
            expected[indicator_name],
            check_names=False,
        )


def test_add_price_action_features_rejects_existing_direction_run_column() -> None:
    data = _prices().assign(candle_direction_run=0)

    with pytest.raises(
        ValueError,
        match="Generated columns already exist in input data: candle_direction_run",
    ):
        add_price_action_features(data)


def _prices() -> pd.DataFrame:
    periods = 5
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * periods,
            ["AAA.ST"] * periods,
            pd.date_range("2026-01-01", periods=periods, freq="D"),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    return pd.DataFrame(
        {
            "open": [98.0, 50.0, 51.0, 53.0, 52.0],
            "high": [102.0, 52.0, 54.0, 54.0, 53.0],
            "low": [97.0, 49.0, 50.0, 51.0, 50.0],
            "close": [100.0, 51.0, 53.0, 52.0, 51.0],
            "adjusted_close": [50.0, 51.0, 53.0, 52.0, 51.0],
        },
        index=index,
    )
