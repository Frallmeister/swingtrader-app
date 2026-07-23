import pandas as pd
import pytest

from swingtrader.data.features.market_structure import (
    add_market_structure_features,
    zigzag_features,
)

_LEVEL_COLUMNS = [
    "market_structure_close_to_prior_high_atr",
    "market_structure_close_to_prior_low_atr",
    "market_structure_breakout_high_strength",
    "market_structure_breakout_low_strength",
    "market_structure_failed_breakout_high_strength",
    "market_structure_failed_breakout_low_strength",
]


def test_zigzag_features_measure_confirmed_swing_level_interactions() -> None:
    result = zigzag_features(
        _indexed_prices(),
        deviation=5.0,
        pivot_legs=2,
        atr_length=1,
    ).reset_index(drop=True)

    assert pd.isna(result.loc[3, "market_structure_close_to_prior_high_atr"])
    assert result.loc[4, "market_structure_close_to_prior_high_atr"] == pytest.approx(-0.5)
    assert result.loc[4, "market_structure_close_to_prior_low_atr"] == pytest.approx(2.0)
    assert result.loc[6, "market_structure_breakout_high_strength"] == pytest.approx(0.5)
    assert result.loc[6, "market_structure_failed_breakout_high_strength"] == pytest.approx(0.0)


def test_zigzag_features_measure_failed_breaks_of_confirmed_levels() -> None:
    high_failure = _prices().copy()
    high_failure.loc[5, ["high", "close"]] = [111.0, 110.0]
    high_result = zigzag_features(
        _indexed(high_failure),
        deviation=5.0,
        pivot_legs=2,
        atr_length=1,
    ).reset_index(drop=True)

    assert high_result.loc[5, "market_structure_close_to_prior_high_atr"] == pytest.approx(0.0)
    assert high_result.loc[5, "market_structure_breakout_high_strength"] == pytest.approx(0.0)
    assert high_result.loc[5, "market_structure_failed_breakout_high_strength"] == pytest.approx(
        0.5
    )

    low_failure = _prices().copy()
    low_failure.loc[5, ["low", "close"]] = [99.0, 100.0]
    low_result = zigzag_features(
        _indexed(low_failure),
        deviation=5.0,
        pivot_legs=2,
        atr_length=1,
    ).reset_index(drop=True)

    assert low_result.loc[5, "market_structure_close_to_prior_low_atr"] == pytest.approx(0.0)
    assert low_result.loc[5, "market_structure_breakout_low_strength"] == pytest.approx(0.0)
    assert low_result.loc[5, "market_structure_failed_breakout_low_strength"] == pytest.approx(0.5)


def test_zigzag_level_interactions_do_not_change_when_future_rows_are_appended() -> None:
    prices = _indexed_prices()
    full_result = zigzag_features(
        prices,
        deviation=5.0,
        pivot_legs=2,
        atr_length=1,
    )

    for stop in range(1, len(prices) + 1):
        prefix_result = zigzag_features(
            prices.iloc[:stop],
            deviation=5.0,
            pivot_legs=2,
            atr_length=1,
        )
        pd.testing.assert_series_equal(
            prefix_result.loc[:, _LEVEL_COLUMNS].iloc[-1],
            full_result.loc[:, _LEVEL_COLUMNS].iloc[stop - 1],
            check_names=False,
        )


def test_add_market_structure_features_forwards_atr_length() -> None:
    prices = _indexed_prices()

    result = add_market_structure_features(
        prices,
        zigzag_deviation=5.0,
        zigzag_pivot_legs=2,
        zigzag_atr_length=1,
    )
    expected = zigzag_features(
        prices,
        deviation=5.0,
        pivot_legs=2,
        atr_length=1,
    )

    pd.testing.assert_frame_equal(result[expected.columns], expected)


def test_zigzag_features_reject_invalid_atr_length() -> None:
    with pytest.raises(ValueError, match="0"):
        zigzag_features(
            _indexed_prices(),
            deviation=5.0,
            pivot_legs=2,
            atr_length=0,
        )


def _indexed_prices() -> pd.DataFrame:
    return _indexed(_prices())


def _indexed(prices: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    return prices.assign(
        provider="yfinance",
        ticker="AAA.ST",
        trading_date=dates,
    ).set_index(["provider", "ticker", "trading_date"])


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": [106.0, 105.0, 108.0, 110.0, 109.0, 108.0, 112.0, 115.0],
            "low": [103.0, 100.0, 104.0, 107.0, 107.0, 106.0, 109.0, 112.0],
            "close": [104.0, 102.0, 106.0, 109.0, 108.0, 107.0, 111.0, 114.0],
        }
    )
