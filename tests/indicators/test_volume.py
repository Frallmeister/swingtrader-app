import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators.volume import mfi


def test_mfi_returns_expected_values_for_mixed_price_changes() -> None:
    # high == low == close makes the typical price equal to the close, so the
    # money flow is driven purely by the hand-checked price changes below.
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 9.5, 12.0, 8.0],
            "low": [10.0, 11.0, 9.5, 12.0, 8.0],
            "close": [10.0, 11.0, 9.5, 12.0, 8.0],
            "volume": [100, 100, 100, 100, 100],
        }
    )

    result = mfi(frame, length=2)

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            100 * 1100 / 2050,
            100 * 1200 / 2150,
            60.0,
        ],
        name="mfi",
    )

    pd.testing.assert_series_equal(result, expected, check_exact=False)


def test_mfi_is_100_without_negative_flow_and_0_without_positive_flow() -> None:
    rising = pd.DataFrame(
        {
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [100, 100, 100, 100, 100],
        }
    )
    falling = pd.DataFrame(
        {
            "high": [5.0, 4.0, 3.0, 2.0, 1.0],
            "low": [5.0, 4.0, 3.0, 2.0, 1.0],
            "close": [5.0, 4.0, 3.0, 2.0, 1.0],
            "volume": [100, 100, 100, 100, 100],
        }
    )

    rising_mfi = mfi(rising, length=2)
    falling_mfi = mfi(falling, length=2)

    assert (rising_mfi.dropna() == 100.0).all()
    assert (falling_mfi.dropna() == 0.0).all()
    # The first ``length`` rows stay missing until the trailing window is full.
    assert rising_mfi.iloc[:2].isna().all()
    assert rising_mfi.notna().sum() == 3


def test_mfi_leaves_flat_window_missing() -> None:
    flat = pd.DataFrame(
        {
            "high": [50.0, 50.0, 50.0, 50.0],
            "low": [50.0, 50.0, 50.0, 50.0],
            "close": [50.0, 50.0, 50.0, 50.0],
            "volume": [100, 100, 100, 100],
        }
    )

    result = mfi(flat, length=2)

    assert result.isna().all()


def test_mfi_stays_within_bounds() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 13.0, 11.0, 14.0, 12.0, 15.0, 13.0, 16.0],
            "low": [8.0, 9.0, 6.0, 8.0, 5.0, 7.0, 4.0, 6.0],
            "close": [9.0, 12.0, 7.0, 13.0, 6.0, 14.0, 5.0, 15.0],
            "volume": [100, 250, 80, 300, 120, 400, 90, 500],
        }
    )

    result = mfi(frame, length=3).dropna()

    assert ((result >= 0.0) & (result <= 100.0)).all()


def test_mfi_allows_non_temporal_index_and_preserves_row_order() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 15.0, 13.0],
            "low": [9.0, 13.0, 11.0],
            "close": [10.0, 14.0, 12.0],
            "volume": [100, 100, 100],
        },
        index=pd.Index([2, 0, 1]),
    )

    result = mfi(frame, length=1)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_mfi_requires_high_low_close_volume() -> None:
    frame = _ohlc().drop(columns="volume")

    with pytest.raises(ValueError, match="Missing required columns"):
        mfi(frame, length=2)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_mfi_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mfi(
            _ohlc(),
            length=length,  # type: ignore[arg-type]
        )


def test_mfi_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = mfi(prices, length=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    # AAA.ST's typical price rises every day, so its warmed-up MFI is a pure 100,
    # isolated from BBB.ST.
    aaa_mfi = result.loc[("yfinance", "AAA.ST")]
    assert (aaa_mfi.dropna() == 100.0).all()
    assert aaa_mfi.isna().sum() == 2
    # BBB.ST is flat and isolated, so its typical price never changes.
    assert result.loc[("yfinance", "BBB.ST")].isna().all()


def _ohlc() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"]).loc[("yfinance", "AAA.ST")]


def _indexed_prices() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"])


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance"] * 8,
            "ticker": [
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
            ],
            "trading_date": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                ]
            ).date,
            "high": [11.0, 13.0, 15.0, 17.0, 100.0, 100.0, 100.0, 100.0],
            "low": [9.0, 11.0, 13.0, 15.0, 100.0, 100.0, 100.0, 100.0],
            "close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
            "adjusted_close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1100, 1200, 1300, 500, 500, 500, 500],
        }
    )
