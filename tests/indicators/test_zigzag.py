import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators.market_structure import zigzag


def test_zigzag_returns_expected_pivots_returns_and_bars() -> None:
    prices = _prices()

    result = zigzag(prices, deviation=5.0, pivot_legs=2)

    expected = pd.DataFrame(
        {
            "zigzag_price": [
                np.nan,
                100.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                115.0,
                np.nan,
                100.0,
                np.nan,
            ],
            "zigzag_direction": pd.Series(
                [0, -1, 0, 0, 0, 0, 0, 1, 0, -1, 0],
                dtype="int8",
            ),
            "zigzag_return": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0.15,
                np.nan,
                100.0 / 115.0 - 1.0,
                np.nan,
            ],
            "zigzag_bars": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                6.0,
                np.nan,
                2.0,
                np.nan,
            ],
        },
        index=prices.index,
    )

    pd.testing.assert_frame_equal(result, expected, check_exact=False)


def test_zigzag_uses_high_for_highs_and_low_for_lows() -> None:
    prices = _prices()

    result = zigzag(prices, deviation=5.0, pivot_legs=2)

    assert result.loc[1, "zigzag_price"] == prices.loc[1, "low"]
    assert result.loc[7, "zigzag_price"] == prices.loc[7, "high"]
    assert result.loc[9, "zigzag_price"] == prices.loc[9, "low"]


def test_zigzag_accepts_exact_downward_deviation_relative_to_last_high() -> None:
    prices = pd.DataFrame(
        {
            "high": [99.0, 100.0, 99.0, 97.0, 98.0],
            "low": [98.0, 99.0, 97.0, 95.0, 96.0],
        }
    )

    result = zigzag(prices, deviation=5.0, pivot_legs=2)

    assert result.loc[1, "zigzag_direction"] == 1
    assert result.loc[3, "zigzag_direction"] == -1
    assert result.loc[3, "zigzag_return"] == pytest.approx(-0.05)


def test_zigzag_uses_absolute_deviation_for_negative_prices() -> None:
    prices = pd.DataFrame(
        {
            "high": [-11.0, -10.0, -11.0, -14.0, -13.0],
            "low": [-12.0, -11.0, -13.0, -15.0, -14.0],
        }
    )

    result = zigzag(prices, deviation=50.0, pivot_legs=2)

    assert result.loc[1, "zigzag_direction"] == 1
    assert result.loc[3, "zigzag_direction"] == -1
    assert result.loc[3, "zigzag_return"] == pytest.approx(0.5)


def test_zigzag_ignores_small_reversal_and_replaces_same_direction_endpoint() -> None:
    result = zigzag(_prices(), deviation=5.0, pivot_legs=2)

    # The low at index 5 is only 3.64% below the high at index 3 and is ignored.
    assert result.loc[5, "zigzag_direction"] == 0

    # The later, higher pivot at index 7 replaces the retained high at index 3.
    assert result.loc[3, "zigzag_direction"] == 0
    assert result.loc[7, "zigzag_direction"] == 1


def test_zigzag_uses_floor_division_for_odd_pivot_legs() -> None:
    prices = _prices()

    even = zigzag(prices, deviation=5.0, pivot_legs=2)
    odd = zigzag(prices, deviation=5.0, pivot_legs=3)

    pd.testing.assert_frame_equal(even, odd)


def test_zigzag_retains_first_equal_extreme() -> None:
    prices = pd.DataFrame(
        {
            "high": [9.0, 10.0, 10.0, 9.0],
            "low": [8.0, 9.0, 9.0, 8.0],
        }
    )

    result = zigzag(prices, deviation=0.0, pivot_legs=2)

    assert result.loc[1, "zigzag_direction"] == 1
    assert result.loc[2, "zigzag_direction"] == 0


def test_zigzag_uses_one_high_first_pivot_when_one_bar_is_both_extrema() -> None:
    prices = pd.DataFrame(
        {
            "high": [10.0, 9.0, 10.0, 13.0, 12.0],
            "low": [9.0, 8.0, 9.0, 7.0, 8.0],
        }
    )

    result = zigzag(prices, deviation=0.0, pivot_legs=2)

    assert result.loc[1, "zigzag_direction"] == -1
    assert result.loc[3, "zigzag_direction"] == 1
    assert result.loc[3, "zigzag_price"] == 13.0


def test_zigzag_preserves_multiindex_and_groups_by_ticker() -> None:
    prices = _multi_ticker_prices()

    result = zigzag(prices, deviation=5.0, pivot_legs=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    for ticker in ["AAA.ST", "BBB.ST"]:
        expected = zigzag(
            prices.loc[("yfinance", ticker)],
            deviation=5.0,
            pivot_legs=2,
        )
        pd.testing.assert_frame_equal(
            result.loc[("yfinance", ticker)],
            expected,
        )


@pytest.mark.parametrize("deviation", [-1.0, True, float("inf"), float("nan")])
def test_zigzag_rejects_invalid_deviation(deviation: object) -> None:
    with pytest.raises(ValueError, match="non-negative finite number"):
        zigzag(
            _prices(),
            deviation=deviation,  # type: ignore[arg-type]
            pivot_legs=2,
        )


@pytest.mark.parametrize("pivot_legs", [0, 1, -1, True, 2.5])
def test_zigzag_rejects_invalid_pivot_legs(pivot_legs: object) -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        zigzag(
            _prices(),
            deviation=5.0,
            pivot_legs=pivot_legs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("column", ["high", "low"])
def test_zigzag_requires_high_and_low(column: str) -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        zigzag(_prices().drop(columns=column), pivot_legs=2)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
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


def _multi_ticker_prices() -> pd.DataFrame:
    aaa = _prices()
    bbb = _prices().mul(2.0)
    dates = pd.date_range("2026-01-01", periods=len(aaa), freq="D")

    aaa.index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * len(aaa),
            ["AAA.ST"] * len(aaa),
            dates,
        ],
        names=["provider", "ticker", "trading_date"],
    )
    bbb.index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * len(bbb),
            ["BBB.ST"] * len(bbb),
            dates,
        ],
        names=["provider", "ticker", "trading_date"],
    )
    return pd.concat([aaa, bbb])
