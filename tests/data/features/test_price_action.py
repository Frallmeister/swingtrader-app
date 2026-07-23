import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features import add_price_action_features
from swingtrader.indicators import candle_geometry, candle_patterns, candle_range_context

_FEATURE_COLUMNS = [
    "candle_signed_body_fraction",
    "candle_upper_wick_fraction",
    "candle_lower_wick_fraction",
    "candle_close_location",
    "candle_range_atr",
    "candle_gap_atr",
    "range_percentile_2",
    "candle_inside_bar",
    "candle_outside_bar",
    "candle_engulfing_strength",
    "candle_lower_rejection_strength",
    "candle_upper_rejection_strength",
    "candle_consecutive_inside_bars",
    "candle_close_to_prior_high_atr_20",
    "candle_close_to_prior_low_atr_20",
    "candle_breakout_high_strength_20",
    "candle_breakout_low_strength_20",
    "candle_failed_break_high_strength_20",
    "candle_failed_break_low_strength_20",
]


def test_add_price_action_features_adds_expected_adjusted_indicator_outputs() -> None:
    data = _prices()
    original = data.copy(deep=True)

    result = add_price_action_features(
        data,
        atr_length=2,
        range_percentile_length=2,
    )

    factor = data["adjusted_close"] / data["close"]
    adjusted_ohlc = data[["open", "high", "low", "close"]].mul(factor, axis=0)
    expected_geometry = candle_geometry(adjusted_ohlc)
    expected_context = candle_range_context(
        adjusted_ohlc,
        atr_length=2,
        percentile_length=2,
    )
    expected_patterns = candle_patterns(adjusted_ohlc, atr_length=2)

    pd.testing.assert_series_equal(
        result["candle_signed_body_fraction"],
        expected_geometry["signed_body_fraction"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["candle_upper_wick_fraction"],
        expected_geometry["upper_wick_fraction"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["candle_lower_wick_fraction"],
        expected_geometry["lower_wick_fraction"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["candle_close_location"],
        expected_geometry["close_location"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["candle_range_atr"],
        expected_context["range_atr"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["candle_gap_atr"],
        expected_context["gap_atr"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        result["range_percentile_2"],
        expected_context["range_percentile"],
        check_names=False,
    )
    pattern_feature_names = {
        "inside_bar": "candle_inside_bar",
        "outside_bar": "candle_outside_bar",
        "engulfing_strength": "candle_engulfing_strength",
        "lower_rejection_strength": "candle_lower_rejection_strength",
        "upper_rejection_strength": "candle_upper_rejection_strength",
        "consecutive_inside_bars": "candle_consecutive_inside_bars",
    }
    for indicator_name, feature_name in pattern_feature_names.items():
        pd.testing.assert_series_equal(
            result[feature_name],
            expected_patterns[indicator_name],
            check_names=False,
        )
    assert result.columns[-len(_FEATURE_COLUMNS) :].tolist() == _FEATURE_COLUMNS
    pd.testing.assert_index_equal(result.index, data.index)
    pd.testing.assert_frame_equal(data, original)


def test_add_price_action_features_does_not_change_when_future_rows_are_appended() -> None:
    data = _prices()
    historical = add_price_action_features(
        data.iloc[:-1],
        atr_length=2,
        range_percentile_length=2,
    )
    extended = add_price_action_features(
        data,
        atr_length=2,
        range_percentile_length=2,
    ).iloc[:-1]

    pd.testing.assert_frame_equal(historical, extended)


def test_add_price_action_features_removes_split_discontinuity_from_gap() -> None:
    index = _index(periods=3)
    data = pd.DataFrame(
        {
            "open": [98.0, 50.0, 51.0],
            "high": [102.0, 52.0, 53.0],
            "low": [97.0, 49.0, 50.0],
            "close": [100.0, 51.0, 52.0],
            "adjusted_close": [50.0, 51.0, 52.0],
        },
        index=index,
    )

    result = add_price_action_features(
        data,
        atr_length=1,
        range_percentile_length=1,
    )
    raw_context = candle_range_context(data, atr_length=1, percentile_length=1)

    assert result["candle_gap_atr"].iloc[1] == pytest.approx(0.0)
    assert raw_context["gap_atr"].iloc[1] == pytest.approx(-10.0)
    assert result["candle_range_atr"].iloc[1] == pytest.approx(1.2)
    assert raw_context["range_atr"].iloc[1] == pytest.approx(10.2)


def test_add_price_action_features_uses_length_in_percentile_column_name() -> None:
    result = add_price_action_features(
        _prices(),
        atr_length=2,
        range_percentile_length=3,
    )

    assert "range_percentile_3" in result.columns
    assert "range_percentile_20" not in result.columns


def test_add_price_action_features_rejects_existing_generated_columns() -> None:
    data = _prices().assign(candle_range_atr=0.0)

    with pytest.raises(
        ValueError,
        match="Generated columns already exist in input data: candle_range_atr",
    ):
        add_price_action_features(data, range_percentile_length=2)


def test_add_price_action_features_requires_canonical_index() -> None:
    data = _prices().reset_index(drop=True)

    with pytest.raises(ValueError, match="Market-price data must use a MultiIndex"):
        add_price_action_features(data, range_percentile_length=2)


def test_add_price_action_features_requires_source_columns() -> None:
    data = _prices().drop(columns="adjusted_close")

    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        add_price_action_features(data, range_percentile_length=2)


def test_add_price_action_features_isolates_tickers() -> None:
    first = _prices()
    second = first.copy()
    second.index = pd.MultiIndex.from_arrays(
        [
            second.index.get_level_values("provider"),
            ["BBB.ST"] * len(second),
            second.index.get_level_values("trading_date"),
        ],
        names=second.index.names,
    )
    combined = pd.concat([first, second]).sort_index()

    result = add_price_action_features(
        combined,
        atr_length=2,
        range_percentile_length=2,
    )

    for ticker in ("AAA.ST", "BBB.ST"):
        ticker_result = result.xs(ticker, level="ticker")
        assert ticker_result["candle_gap_atr"].iloc[:2].isna().all()
        assert ticker_result["range_percentile_2"].iloc[:2].isna().all()
        assert pd.isna(ticker_result["candle_inside_bar"].iloc[0])
        assert pd.isna(ticker_result["candle_consecutive_inside_bars"].iloc[0])


def _prices() -> pd.DataFrame:
    index = _index(periods=5)
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 11.0, 13.0],
            "high": [12.0, 13.0, 14.0, 13.0, 15.0],
            "low": [9.0, 10.0, 11.0, 9.0, 12.0],
            "close": [11.0, 12.0, 13.0, 10.0, 14.0],
            "adjusted_close": [5.5, 6.0, 6.5, 10.0, 14.0],
            "volume": np.arange(1_000.0, 1_005.0),
        },
        index=index,
    )


def _index(*, periods: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * periods,
            ["AAA.ST"] * periods,
            pd.date_range("2026-01-01", periods=periods, freq="D"),
        ],
        names=["provider", "ticker", "trading_date"],
    )
