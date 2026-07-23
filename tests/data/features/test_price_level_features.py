import pandas as pd

from swingtrader.data.features.price_action import add_price_action_features
from swingtrader.indicators import rolling_level_interactions

_LEVEL_FEATURE_NAMES = {
    "close_to_upper_atr": "candle_close_to_prior_high_atr_3",
    "close_to_lower_atr": "candle_close_to_prior_low_atr_3",
    "breakout_high_strength": "candle_breakout_high_strength_3",
    "breakout_low_strength": "candle_breakout_low_strength_3",
    "failed_break_high_strength": "candle_failed_breakout_high_strength_3",
    "failed_break_low_strength": "candle_failed_breakout_low_strength_3",
}


def test_add_price_action_features_adds_adjusted_rolling_level_outputs() -> None:
    data = _prices()

    result = add_price_action_features(
        data,
        atr_length=1,
        range_percentile_length=2,
        breakout_length=3,
    )
    factor = data["adjusted_close"] / data["close"]
    adjusted_ohlc = data[["open", "high", "low", "close"]].mul(factor, axis=0)
    expected = rolling_level_interactions(adjusted_ohlc, length=3, atr_length=1)

    for indicator_name, feature_name in _LEVEL_FEATURE_NAMES.items():
        pd.testing.assert_series_equal(
            result[feature_name],
            expected[indicator_name],
            check_names=False,
        )

    assert "prior_high" not in result.columns
    assert "prior_low" not in result.columns


def test_add_price_action_features_level_outputs_do_not_change_with_future_rows() -> None:
    data = _prices()
    feature_names = list(_LEVEL_FEATURE_NAMES.values())

    historical = add_price_action_features(
        data.iloc[:-1],
        atr_length=1,
        range_percentile_length=2,
        breakout_length=3,
    )
    extended = add_price_action_features(
        data,
        atr_length=1,
        range_percentile_length=2,
        breakout_length=3,
    ).iloc[:-1]

    pd.testing.assert_frame_equal(historical[feature_names], extended[feature_names])


def _prices() -> pd.DataFrame:
    periods = 7
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
            "open": [9.0, 10.0, 11.0, 11.5, 12.5, 9.0, 8.0],
            "high": [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 13.0],
            "low": [8.0, 9.0, 10.0, 9.0, 10.0, 8.0, 7.0],
            "close": [9.0, 10.0, 11.0, 12.5, 13.0, 8.5, 8.5],
            "adjusted_close": [4.5, 5.0, 5.5, 12.5, 13.0, 8.5, 8.5],
        },
        index=index,
    )
