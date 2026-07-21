import math

import pandas as pd
import pytest

from swingtrader.data.features.market_structure import zigzag_features


def test_zigzag_features_calculate_structural_log_changes_and_rates() -> None:
    result = zigzag_features(_market_prices(), deviation=5.0, pivot_legs=2)
    bullish = result.loc[("yfinance", "BULL.ST")].reset_index(drop=True)

    assert bullish.loc[8, "market_structure_low_change"] == pytest.approx(math.log(94.0 / 89.0))
    assert bullish.loc[8, "market_structure_low_rate"] == pytest.approx(math.log(94.0 / 89.0) / 6.0)
    assert bullish.loc[11, "market_structure_high_change"] == pytest.approx(math.log(122.0 / 111.0))
    assert bullish.loc[11, "market_structure_high_rate"] == pytest.approx(
        math.log(122.0 / 111.0) / 6.0
    )
    assert bullish.loc[13, "market_structure_low_change"] == pytest.approx(math.log(99.0 / 94.0))
    assert bullish.loc[13, "market_structure_low_rate"] == pytest.approx(
        math.log(99.0 / 94.0) / 5.0
    )
    assert bullish.loc[16, "market_structure_high_change"] == pytest.approx(math.log(134.0 / 122.0))
    assert bullish.loc[16, "market_structure_high_rate"] == pytest.approx(
        math.log(134.0 / 122.0) / 5.0
    )


def test_structural_changes_wait_for_two_confirmed_pivots_of_each_direction() -> None:
    result = zigzag_features(_market_prices(), deviation=5.0, pivot_legs=2)
    bullish = result.loc[("yfinance", "BULL.ST")].reset_index(drop=True)

    assert bullish.loc[:7, "market_structure_low_change"].isna().all()
    assert bullish.loc[:7, "market_structure_low_rate"].isna().all()
    assert bullish.loc[:10, "market_structure_high_change"].isna().all()
    assert bullish.loc[:10, "market_structure_high_rate"].isna().all()

    assert pd.notna(bullish.loc[8, "market_structure_low_change"])
    assert pd.notna(bullish.loc[11, "market_structure_high_change"])


def test_structural_changes_are_scale_independent_and_ticker_isolated() -> None:
    result = zigzag_features(_market_prices(), deviation=5.0, pivot_legs=2)
    bullish = result.loc[("yfinance", "BULL.ST")].reset_index(drop=True)
    scaled = result.loc[("yfinance", "SCALED.ST")].reset_index(drop=True)
    columns = [
        "market_structure_high_change",
        "market_structure_low_change",
        "market_structure_high_rate",
        "market_structure_low_rate",
    ]

    pd.testing.assert_frame_equal(bullish[columns], scaled[columns])


def test_structural_changes_are_negative_for_lower_highs_and_lower_lows() -> None:
    result = zigzag_features(_market_prices(), deviation=5.0, pivot_legs=2)
    bearish = result.loc[("yfinance", "BEAR.ST")].reset_index(drop=True)

    assert bearish.loc[8, "market_structure_low_change"] == pytest.approx(math.log(119.0 / 129.0))
    assert bearish.loc[8, "market_structure_low_rate"] < 0.0
    assert bearish.loc[11, "market_structure_high_change"] == pytest.approx(math.log(136.0 / 146.0))
    assert bearish.loc[11, "market_structure_high_rate"] < 0.0


def _market_prices() -> pd.DataFrame:
    bullish_close = pd.Series(
        [
            100.0,
            90.0,
            100.0,
            105.0,
            110.0,
            104.0,
            100.0,
            95.0,
            103.0,
            112.0,
            121.0,
            110.0,
            100.0,
            108.0,
            120.0,
            133.0,
            115.0,
            105.0,
            110.0,
        ]
    )
    bearish_close = pd.Series(
        [
            140.0,
            130.0,
            138.0,
            142.0,
            145.0,
            138.0,
            130.0,
            120.0,
            128.0,
            132.0,
            135.0,
            125.0,
            110.0,
            115.0,
            120.0,
            125.0,
            112.0,
            100.0,
            105.0,
        ]
    )
    dates = pd.date_range("2026-01-01", periods=len(bullish_close), freq="D")

    bullish = _instrument_prices(bullish_close, ticker="BULL.ST", dates=dates)
    scaled = bullish.mul(2.0)
    scaled.index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * len(scaled),
            ["SCALED.ST"] * len(scaled),
            dates,
        ],
        names=["provider", "ticker", "trading_date"],
    )
    bearish = _instrument_prices(bearish_close, ticker="BEAR.ST", dates=dates)

    return pd.concat([bearish, bullish, scaled]).sort_index()


def _instrument_prices(
    close: pd.Series,
    *,
    ticker: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": close.to_numpy() + 1.0,
            "low": close.to_numpy() - 1.0,
            "close": close.to_numpy(),
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["yfinance"] * len(close),
                [ticker] * len(close),
                dates,
            ],
            names=["provider", "ticker", "trading_date"],
        ),
    )
