import pandas as pd
import pytest

from swingtrader.data.features.market_structure import (
    add_market_structure_features,
    zigzag_features,
    zigzag_retracement,
    zigzag_swing_return_per_bar,
)


def test_zigzag_features_are_emitted_only_after_pivot_confirmation() -> None:
    prices = _indexed_prices()

    result = zigzag_features(prices, deviation=5.0, pivot_legs=2)
    aaa = result.loc[("yfinance", "AAA.ST")].reset_index(drop=True)

    # The low at position 1 is confirmed at position 2.
    assert pd.isna(aaa.loc[1, "zigzag_last_direction"])
    assert aaa.loc[2, "zigzag_last_direction"] == -1
    assert aaa.loc[2, "zigzag_bars_since_pivot"] == 1.0

    # The high at position 3 is not available until position 4.
    assert aaa.loc[3, "zigzag_last_direction"] == -1
    assert aaa.loc[4, "zigzag_last_direction"] == 1


def test_zigzag_features_keep_intermediate_endpoint_until_replacement_is_confirmed() -> (
    None
):
    result = zigzag_features(
        _indexed_prices(),
        deviation=5.0,
        pivot_legs=2,
    )
    aaa = result.loc[("yfinance", "AAA.ST")].reset_index(drop=True)

    # The high at position 3 remains the point-in-time endpoint through position 7.
    assert aaa.loc[7, "zigzag_last_swing_return"] == pytest.approx(0.10)
    assert aaa.loc[7, "zigzag_last_swing_bars"] == 2.0

    # The higher pivot at position 7 replaces it when confirmed at position 8.
    assert aaa.loc[8, "zigzag_last_swing_return"] == pytest.approx(0.15)
    assert aaa.loc[8, "zigzag_last_swing_bars"] == 6.0


def test_zigzag_features_calculate_return_per_bar_and_retracement() -> None:
    result = zigzag_features(
        _indexed_prices(),
        deviation=5.0,
        pivot_legs=2,
    )
    aaa = result.loc[("yfinance", "AAA.ST")].reset_index(drop=True)

    assert aaa.loc[8, "zigzag_swing_return_per_bar"] == pytest.approx(
        1.15 ** (1.0 / 6.0) - 1.0
    )
    assert aaa.loc[8, "zigzag_retracement"] == pytest.approx(7.0 / 15.0)

    # At position 10 the last confirmed swing is 115 -> 100 and close is 103.
    assert aaa.loc[10, "zigzag_last_swing_return"] == pytest.approx(100.0 / 115.0 - 1.0)
    assert aaa.loc[10, "zigzag_last_swing_bars"] == 2.0
    assert aaa.loc[10, "zigzag_bars_since_pivot"] == 1.0
    assert aaa.loc[10, "zigzag_retracement"] == pytest.approx(0.2)


def test_zigzag_features_do_not_change_when_future_rows_are_appended() -> None:
    prices = _indexed_prices()
    aaa = prices[prices.index.get_level_values("ticker") == "AAA.ST"]
    full_result = zigzag_features(aaa, deviation=5.0, pivot_legs=2)

    for stop in range(1, len(aaa) + 1):
        prefix_result = zigzag_features(
            aaa.iloc[:stop],
            deviation=5.0,
            pivot_legs=2,
        )
        pd.testing.assert_series_equal(
            prefix_result.iloc[-1],
            full_result.iloc[stop - 1],
            check_names=False,
        )


def test_public_single_feature_helpers_match_feature_block() -> None:
    prices = _indexed_prices()
    block = zigzag_features(prices, deviation=5.0, pivot_legs=2)

    pd.testing.assert_series_equal(
        zigzag_retracement(prices, deviation=5.0, pivot_legs=2),
        block["zigzag_retracement"],
    )
    pd.testing.assert_series_equal(
        zigzag_swing_return_per_bar(prices, deviation=5.0, pivot_legs=2),
        block["zigzag_swing_return_per_bar"],
    )


def test_add_market_structure_features_preserves_input_and_appends_block() -> None:
    prices = _indexed_prices()
    original = prices.copy(deep=True)

    result = add_market_structure_features(
        prices,
        zigzag_deviation=5.0,
        zigzag_pivot_legs=2,
    )
    expected_block = zigzag_features(prices, deviation=5.0, pivot_legs=2)

    pd.testing.assert_frame_equal(prices, original)
    pd.testing.assert_index_equal(result.index, prices.index)
    assert list(result.columns) == [*prices.columns, *expected_block.columns]
    pd.testing.assert_frame_equal(result[expected_block.columns], expected_block)


def test_zigzag_features_calculate_each_ticker_independently() -> None:
    prices = _indexed_prices()

    result = zigzag_features(prices, deviation=5.0, pivot_legs=2)

    aaa = result.loc[("yfinance", "AAA.ST")]
    bbb = result.loc[("yfinance", "BBB.ST")]
    pd.testing.assert_frame_equal(
        aaa.reset_index(drop=True),
        bbb.reset_index(drop=True),
    )


def test_zigzag_features_reject_noncanonical_input() -> None:
    with pytest.raises(ValueError, match="MultiIndex with levels"):
        zigzag_features(_prices(), deviation=5.0, pivot_legs=2)


def test_zigzag_features_reject_unsorted_input() -> None:
    prices = _indexed_prices().iloc[[1, 0, *range(2, 22)]]

    with pytest.raises(ValueError, match="must be sorted"):
        zigzag_features(prices, deviation=5.0, pivot_legs=2)


@pytest.mark.parametrize("column", ["high", "low", "close"])
def test_zigzag_features_require_price_columns(column: str) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        zigzag_features(
            _indexed_prices().drop(columns=column),
            deviation=5.0,
            pivot_legs=2,
        )


def _indexed_prices() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"])


def _prices() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "high": [
                106.0,
                105.0,
                108.0,
                110.0,
                109.0,
                108.0,
                112.0,
                115.0,
                111.0,
                103.0,
                105.0,
            ],
            "low": [
                103.0,
                100.0,
                104.0,
                107.0,
                107.0,
                106.0,
                109.0,
                112.0,
                105.0,
                100.0,
                102.0,
            ],
            "close": [
                104.0,
                102.0,
                106.0,
                109.0,
                108.0,
                107.0,
                111.0,
                114.0,
                108.0,
                101.0,
                103.0,
            ],
        }
    )
    dates = pd.date_range("2026-01-01", periods=len(base), freq="D")

    aaa = base.assign(
        provider="yfinance",
        ticker="AAA.ST",
        trading_date=dates,
    )
    bbb = base.assign(
        high=base["high"] * 2.0,
        low=base["low"] * 2.0,
        close=base["close"] * 2.0,
        provider="yfinance",
        ticker="BBB.ST",
        trading_date=dates,
    )
    return pd.concat([aaa, bbb], ignore_index=True)
