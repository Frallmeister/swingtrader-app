import math

import pandas as pd
import pytest

from swingtrader.data.features.market_structure import (
    add_market_structure_features,
    zigzag_features,
)


def test_consistency_is_one_for_persistently_rising_highs_and_lows() -> None:
    bullish = _instrument_prices(
        highs=[110.0, 120.0, 130.0, 140.0],
        lows=[90.0, 95.0, 100.0, 105.0, 110.0],
        ticker="BULL.ST",
    )
    result = _consistency_features(bullish)

    assert result.loc[14, "market_structure_low_consistency"] == pytest.approx(1.0)
    assert result.loc[16, "market_structure_high_consistency"] == pytest.approx(1.0)


def test_consistency_is_minus_one_for_falling_highs_and_lows() -> None:
    bearish = _instrument_prices(
        highs=[150.0, 140.0, 130.0, 120.0],
        lows=[130.0, 120.0, 110.0, 100.0, 90.0],
        ticker="BEAR.ST",
    )
    result = _consistency_features(bearish)

    assert result.loc[14, "market_structure_low_consistency"] == pytest.approx(-1.0)
    assert result.loc[16, "market_structure_high_consistency"] == pytest.approx(-1.0)


def test_consistency_uses_kendall_tau_for_non_monotonic_pivots() -> None:
    mixed = _instrument_prices(
        highs=[110.0, 130.0, 120.0, 140.0],
        lows=[90.0, 95.0, 100.0, 105.0, 110.0],
        ticker="MIXED.ST",
    )
    result = _consistency_features(mixed)

    # Five high pairs are concordant and one is discordant.
    assert result.loc[16, "market_structure_high_consistency"] == pytest.approx(2.0 / 3.0)


def test_consistency_uses_tau_b_for_tied_prices() -> None:
    tied = _instrument_prices(
        highs=[110.0, 120.0, 120.0, 130.0],
        lows=[90.0, 95.0, 100.0, 105.0, 110.0],
        ticker="TIED.ST",
    )
    result = _consistency_features(tied)

    # Five high pairs are concordant and one is tied.
    expected = 5.0 / math.sqrt(5.0 * 6.0)
    assert result.loc[16, "market_structure_high_consistency"] == pytest.approx(expected)


def test_consistency_waits_for_the_configured_number_of_pivots() -> None:
    bullish = _instrument_prices(
        highs=[110.0, 120.0, 130.0, 140.0],
        lows=[90.0, 95.0, 100.0, 105.0, 110.0],
        ticker="BULL.ST",
    )
    result = _consistency_features(bullish)

    assert result.loc[:13, "market_structure_low_consistency"].isna().all()
    assert result.loc[:15, "market_structure_high_consistency"].isna().all()


def test_add_market_structure_features_forwards_consistency_pivots() -> None:
    prices = _instrument_prices(
        highs=[110.0, 120.0, 130.0, 140.0],
        lows=[90.0, 95.0, 100.0, 105.0, 110.0],
        ticker="BULL.ST",
    )
    result = add_market_structure_features(
        prices,
        zigzag_deviation=0.0,
        zigzag_pivot_legs=2,
        zigzag_consistency_pivots=3,
    )
    expected = zigzag_features(
        prices,
        deviation=0.0,
        pivot_legs=2,
        consistency_pivots=3,
    )

    pd.testing.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize("consistency_pivots", [True, 1, 1.5])
def test_consistency_pivots_must_be_an_integer_at_least_two(
    consistency_pivots: object,
) -> None:
    prices = _instrument_prices(
        highs=[110.0, 120.0, 130.0, 140.0],
        lows=[90.0, 95.0, 100.0, 105.0, 110.0],
        ticker="BULL.ST",
    )

    with pytest.raises(ValueError, match="consistency_pivots must be an integer"):
        zigzag_features(
            prices,
            deviation=0.0,
            pivot_legs=2,
            consistency_pivots=consistency_pivots,  # type: ignore[arg-type]
        )


def _consistency_features(prices: pd.DataFrame) -> pd.DataFrame:
    return zigzag_features(
        prices,
        deviation=0.0,
        pivot_legs=2,
        consistency_pivots=4,
    ).reset_index(drop=True)


def _instrument_prices(
    *,
    highs: list[float],
    lows: list[float],
    ticker: str,
) -> pd.DataFrame:
    pivots = [value for pair in zip(lows[:-1], highs, strict=True) for value in pair]
    pivots.append(lows[-1])

    close = [(pivots[0] + pivots[1]) / 2.0]
    for current, following in zip(pivots, pivots[1:], strict=False):
        close.extend([current, (current + following) / 2.0])
    close.extend([pivots[-1], (pivots[-2] + pivots[-1]) / 2.0])

    dates = pd.date_range("2026-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "high": pd.Series(close, dtype="float64").add(1.0).to_numpy(),
            "low": pd.Series(close, dtype="float64").sub(1.0).to_numpy(),
            "close": close,
        },
        index=pd.MultiIndex.from_arrays(
            [["yfinance"] * len(close), [ticker] * len(close), dates],
            names=["provider", "ticker", "trading_date"],
        ),
    )
