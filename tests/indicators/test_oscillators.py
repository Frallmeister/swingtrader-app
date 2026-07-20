import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators.oscillators import rsi, stochastic_oscillator


def test_rsi_is_100_without_losses_and_0_without_gains() -> None:
    rising = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="adjusted_close")
    falling = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], name="adjusted_close")

    rising_rsi = rsi(rising, length=2)
    falling_rsi = rsi(falling, length=2)

    assert (rising_rsi.dropna() == 100.0).all()
    assert (falling_rsi.dropna() == 0.0).all()
    # The first ``length`` rows stay missing until the smoothing window is full.
    assert rising_rsi.iloc[:2].isna().all()
    assert rising_rsi.notna().sum() == 3


def test_rsi_leaves_flat_series_missing() -> None:
    flat = pd.Series([50.0, 50.0, 50.0, 50.0], name="adjusted_close")

    result = rsi(flat, length=2)

    assert result.isna().all()


def test_rsi_stays_within_bounds() -> None:
    values = pd.Series([10.0, 11.0, 9.5, 12.0, 8.0, 13.0, 11.5, 14.0], name="adjusted_close")

    result = rsi(values, length=3).dropna()

    assert ((result >= 0.0) & (result <= 100.0)).all()


def test_rsi_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    result = rsi(values, length=1)

    pd.testing.assert_index_equal(result.index, values.index)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_rsi_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        rsi(
            _prices()["adjusted_close"],
            length=length,  # type: ignore[arg-type]
        )


def test_rsi_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = rsi(close, length=2)

    pd.testing.assert_index_equal(result.index, close.index)
    # AAA.ST rises monotonically, so its warmed-up RSI is a pure 100.
    aaa_rsi = result.loc[("yfinance", "AAA.ST")]
    assert (aaa_rsi.dropna() == 100.0).all()
    assert aaa_rsi.isna().sum() == 2
    # BBB.ST is flat and isolated, so it has no gains or losses.
    assert result.loc[("yfinance", "BBB.ST")].isna().all()


def test_rsi_returns_expected_values_for_mixed_price_changes() -> None:
    values = pd.Series(
        [10.0, 11.0, 9.5, 12.0, 8.0, 13.0, 11.5, 14.0],
        name="adjusted_close",
    )

    result = rsi(values, length=3)

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            79.3103448275862,
            35.3846153846154,
            68.3018867924528,
            55.5640832853026,
            69.6937969374878,
        ],
        name="rsi",
    )

    pd.testing.assert_series_equal(result, expected, check_exact=False)


def test_stochastic_oscillator_returns_dataframe_with_expected_columns_and_values() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 14.0, 13.0],
            "low": [9.0, 10.0, 8.0, 11.0, 7.0],
            "close": [10.0, 11.5, 9.0, 13.0, 8.0],
        }
    )

    result = stochastic_oscillator(frame, k_length=3, k_smoothing=1, d_length=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["stochastic_k", "stochastic_d"]
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_series_equal(
        result["stochastic_k"],
        pd.Series(
            [
                np.nan,
                np.nan,
                20.0,
                100 * 5 / 6,
                100 * 1 / 7,
            ],
            name="stochastic_k",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["stochastic_d"],
        result["stochastic_k"].rolling(window=2, min_periods=2).mean().rename("stochastic_d"),
        check_exact=False,
    )


def test_stochastic_oscillator_smooths_slow_k_with_k_smoothing() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 14.0, 13.0],
            "low": [9.0, 10.0, 8.0, 11.0, 7.0],
            "close": [10.0, 11.5, 9.0, 13.0, 8.0],
        }
    )

    fast = stochastic_oscillator(frame, k_length=3, k_smoothing=1, d_length=1)
    slow = stochastic_oscillator(frame, k_length=3, k_smoothing=2, d_length=1)

    # The slow %K is the fast (raw) %K smoothed with a two-row simple average.
    pd.testing.assert_series_equal(
        slow["stochastic_k"],
        fast["stochastic_k"].rolling(window=2, min_periods=2).mean().rename("stochastic_k"),
        check_exact=False,
    )


def test_stochastic_oscillator_is_100_at_range_high_and_0_at_range_low() -> None:
    # The close sits at the highest high every row, so %K tops out at 100.
    top = pd.DataFrame(
        {
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [2.0, 3.0, 4.0, 5.0],
        }
    )
    # The close sits at the lowest low every row, so %K bottoms out at 0.
    bottom = pd.DataFrame(
        {
            "high": [5.0, 5.0, 5.0, 5.0],
            "low": [4.0, 3.0, 2.0, 1.0],
            "close": [4.0, 3.0, 2.0, 1.0],
        }
    )

    top_k = stochastic_oscillator(top, k_length=2, k_smoothing=1, d_length=1)["stochastic_k"]
    bottom_k = stochastic_oscillator(bottom, k_length=2, k_smoothing=1, d_length=1)["stochastic_k"]

    assert (top_k.dropna() == 100.0).all()
    assert (bottom_k.dropna() == 0.0).all()
    # The first ``k_length - 1`` rows stay missing until the window is full.
    assert top_k.iloc[:1].isna().all()
    assert top_k.notna().sum() == 3


def test_stochastic_oscillator_leaves_flat_window_missing() -> None:
    flat = pd.DataFrame(
        {
            "high": [50.0, 50.0, 50.0, 50.0],
            "low": [50.0, 50.0, 50.0, 50.0],
            "close": [50.0, 50.0, 50.0, 50.0],
        }
    )

    result = stochastic_oscillator(flat, k_length=2, k_smoothing=1, d_length=1)

    assert result["stochastic_k"].isna().all()
    assert result["stochastic_d"].isna().all()


def test_stochastic_oscillator_stays_within_bounds() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 13.0, 11.0, 14.0, 12.0, 15.0, 13.0, 16.0],
            "low": [8.0, 9.0, 6.0, 8.0, 5.0, 7.0, 4.0, 6.0],
            "close": [9.0, 12.0, 7.0, 13.0, 6.0, 14.0, 5.0, 15.0],
        }
    )

    result = stochastic_oscillator(frame, k_length=3)

    stochastic_k = result["stochastic_k"].dropna()
    stochastic_d = result["stochastic_d"].dropna()
    assert ((stochastic_k >= 0.0) & (stochastic_k <= 100.0)).all()
    assert ((stochastic_d >= 0.0) & (stochastic_d <= 100.0)).all()


def test_stochastic_oscillator_allows_non_temporal_index_and_preserves_row_order() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 15.0, 13.0],
            "low": [9.0, 13.0, 11.0],
            "close": [10.0, 14.0, 12.0],
        },
        index=pd.Index([2, 0, 1]),
    )

    result = stochastic_oscillator(frame, k_length=1, k_smoothing=1, d_length=1)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_stochastic_oscillator_requires_high_low_close() -> None:
    frame = _ohlc().drop(columns="close")

    with pytest.raises(ValueError, match="Missing required columns"):
        stochastic_oscillator(frame, k_length=2)


@pytest.mark.parametrize(
    ("k_length", "k_smoothing", "d_length"),
    [(0, 3, 3), (14, 0, 3), (14, 3, 0), (True, 3, 3), (14, 3, 1.5)],
)
def test_stochastic_oscillator_rejects_invalid_lengths(
    k_length: object, k_smoothing: object, d_length: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        stochastic_oscillator(
            _ohlc(),
            k_length=k_length,  # type: ignore[arg-type]
            k_smoothing=k_smoothing,  # type: ignore[arg-type]
            d_length=d_length,  # type: ignore[arg-type]
        )


def test_stochastic_oscillator_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = stochastic_oscillator(prices, k_length=2, k_smoothing=1, d_length=1)

    assert list(result.columns) == ["stochastic_k", "stochastic_d"]
    pd.testing.assert_index_equal(result.index, prices.index)
    # AAA.ST's close sits three-quarters up each two-day range, so its warmed-up
    # %K is a constant 75, isolated from BBB.ST.
    aaa_k = result.loc[("yfinance", "AAA.ST"), "stochastic_k"]
    assert (aaa_k.dropna() == 75.0).all()
    assert aaa_k.isna().sum() == 1
    # BBB.ST is flat and isolated, so every window has no range.
    assert result.loc[("yfinance", "BBB.ST"), "stochastic_k"].isna().all()


def _ohlc() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"]).loc[("yfinance", "AAA.ST")]


def _multi_ticker_close() -> pd.Series:
    return _prices().set_index(["provider", "ticker", "trading_date"])["adjusted_close"]


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
