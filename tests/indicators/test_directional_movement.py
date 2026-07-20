import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators._smoothing import wilder_moving_average
from swingtrader.indicators.directional_movement import adx


def test_adx_returns_expected_columns_and_values() -> None:
    frame = _ohlc()

    result = adx(frame, length=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["adx", "plus_di", "minus_di"]
    pd.testing.assert_index_equal(result.index, frame.index)

    # A strictly rising ticker has only positive directional movement, so
    # minus_di collapses to zero and plus_di follows the smoothed +DM / TR ratio.
    pd.testing.assert_series_equal(
        result["plus_di"].reset_index(drop=True),
        pd.Series(
            [
                np.nan,
                40.0,
                100 * 1.5 / 2.75,
                100 * 1.75 / 2.875,
            ],
            name="plus_di",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["minus_di"].reset_index(drop=True),
        pd.Series([np.nan, 0.0, 0.0, 0.0], name="minus_di"),
        check_exact=False,
    )
    # With no negative directional movement DX is pinned at 100, so its Wilder
    # smoothing is 100 once two DX observations exist.
    assert result["adx"].dropna().eq(100.0).all()


def test_adx_is_bounded_and_directional() -> None:
    frame = _ohlc()

    result = adx(frame, length=2)

    populated = result.dropna()
    assert populated["plus_di"].between(0, 100).all()
    assert populated["minus_di"].between(0, 100).all()
    assert populated["adx"].between(0, 100).all()
    # A rising ticker trends up, so the positive indicator dominates.
    assert (populated["plus_di"] > populated["minus_di"]).all()


def test_adx_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = adx(prices, length=2)

    assert list(result.columns) == ["adx", "plus_di", "minus_di"]
    pd.testing.assert_index_equal(result.index, prices.index)
    # A constant BBB.ST price has zero True Range, so the directional ratios are
    # undefined (NA), isolated from AAA.ST's rising trend.
    assert result.loc[("yfinance", "BBB.ST"), "plus_di"].isna().all()
    assert result.loc[("yfinance", "BBB.ST"), "minus_di"].isna().all()
    assert result.loc[("yfinance", "BBB.ST"), "adx"].isna().all()
    assert result.loc[("yfinance", "AAA.ST"), "plus_di"].notna().sum() == 3


def test_adx_allows_non_temporal_index_and_preserves_row_order() -> None:
    frame = _ohlc().set_axis(pd.Index([2, 0, 1, 3]))

    result = adx(frame, length=2)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_adx_requires_high_low_close() -> None:
    frame = _ohlc().drop(columns="close")

    with pytest.raises(ValueError, match="Missing required columns"):
        adx(frame, length=2)


@pytest.mark.parametrize("length", [0, -1, True, 1.5, "2"])
def test_adx_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        adx(_ohlc(), length=length)  # type: ignore[arg-type]


def test_adx_matches_wilder_primitives() -> None:
    prices = _indexed_prices()

    result = adx(prices, length=2)

    aaa = prices.loc[("yfinance", "AAA.ST")]
    high, low, close = aaa["high"], aaa["low"], aaa["close"]
    up_move = high.diff()
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    expected_plus_di = (
        100 * wilder_moving_average(plus_dm, length=2) / wilder_moving_average(true_range, length=2)
    )

    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST"), "plus_di"].reset_index(drop=True),
        expected_plus_di.reset_index(drop=True).rename("plus_di"),
        check_exact=False,
    )


def test_adx_returns_expected_values_for_mixed_directional_movement() -> None:
    # High and low oscillate up and down every row, so +DM and -DM alternate in
    # dominance and both directional indicators stay populated. ADX then settles
    # into a low "weak trend" reading rather than being pinned at 100 the way a
    # one-directional series leaves it.
    frame = pd.DataFrame(
        {
            "high": [10.0, 13.0, 11.0, 14.0, 12.0, 15.0, 13.0, 16.0],
            "low": [8.0, 9.0, 6.0, 8.0, 5.0, 7.0, 4.0, 6.0],
            "close": [9.0, 12.0, 7.0, 13.0, 6.0, 14.0, 5.0, 15.0],
        }
    )

    result = adx(frame, length=3)

    expected = pd.DataFrame(
        {
            "adx": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                23.74269005847953,
                24.517243609286506,
                23.011495739524342,
                22.836545160330456,
            ],
            "plus_di": [
                np.nan,
                np.nan,
                17.64705882352941,
                29.770992366412212,
                16.317991631799163,
                23.679525222551927,
                13.75862068965517,
                19.28232835516591,
            ],
            "minus_di": [
                np.nan,
                np.nan,
                26.470588235294112,
                13.740458015267176,
                24.476987447698743,
                13.887240356083087,
                20.637931034482758,
                12.202456802079617,
            ],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_exact=False,
    )


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
        }
    )
