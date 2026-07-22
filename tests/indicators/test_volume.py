import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators.volume import mfi, turnover, turnover_zscore


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


def test_turnover_calculates_adjusted_close_times_volume() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 12.5, 8.0],
            "volume": [100, 200, 50],
        }
    )

    result = turnover(frame)

    expected = pd.Series(
        [1000.0, 2500.0, 400.0],
        name="turnover",
    )
    pd.testing.assert_series_equal(result, expected)


def test_turnover_applies_log1p_when_requested() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 12.5, 8.0],
            "volume": [100, 200, 0],
        }
    )

    result = turnover(frame, log=True)

    expected = pd.Series(
        np.log1p([1000.0, 2500.0, 0.0]),
        name="turnover",
    )
    pd.testing.assert_series_equal(result, expected)


def test_turnover_requires_adjusted_close_and_volume() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns: volume"):
        turnover(frame)


@pytest.mark.parametrize("log", [0, 1, "true", None])
def test_turnover_rejects_non_boolean_log(log: object) -> None:
    with pytest.raises(ValueError, match="log parameter must be a boolean"):
        turnover(
            pd.DataFrame(
                {
                    "adjusted_close": [10.0],
                    "volume": [100],
                }
            ),
            log=log,  # type: ignore[arg-type]
        )


def test_turnover_zscore_uses_only_preceding_observations() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 10.0, 10.0, 10.0],
            "volume": [10.0, 20.0, 30.0, 100.0],
        }
    )

    result = turnover_zscore(frame, length=4)

    prior_turnover = np.array([100.0, 200.0, 300.0])
    expected_last = (1000.0 - np.median(prior_turnover)) / np.std(prior_turnover, ddof=0)

    expected = pd.Series(
        [np.nan, np.nan, np.nan, expected_last],
        name="turnover_zscore",
    )
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
    )


def test_turnover_zscore_applies_log_transform_before_normalization() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 10.0, 10.0, 10.0],
            "volume": [10.0, 20.0, 30.0, 100.0],
        }
    )

    result = turnover_zscore(frame, length=4, log=True)

    transformed = np.log1p([100.0, 200.0, 300.0, 1000.0])
    expected_last = (transformed[-1] - np.median(transformed[:-1])) / np.std(
        transformed[:-1], ddof=0
    )

    expected = pd.Series(
        [np.nan, np.nan, np.nan, expected_last],
        name="turnover_zscore",
    )
    pd.testing.assert_series_equal(
        result,
        expected,
        check_exact=False,
    )


def test_turnover_zscore_does_not_include_current_value_in_reference_window() -> None:
    baseline = pd.DataFrame(
        {
            "adjusted_close": [10.0, 10.0, 10.0, 10.0],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )
    outlier = baseline.copy()
    outlier.loc[3, "volume"] = 1_000_000.0

    baseline_result = turnover_zscore(baseline, length=4)
    outlier_result = turnover_zscore(outlier, length=4)

    prior_turnover = np.array([100.0, 200.0, 300.0])
    prior_std = np.std(prior_turnover, ddof=0)

    baseline_expected = (400.0 - 200.0) / prior_std
    outlier_expected = (10_000_000.0 - 200.0) / prior_std

    assert baseline_result.iloc[-1] == pytest.approx(baseline_expected)
    assert outlier_result.iloc[-1] == pytest.approx(outlier_expected)


def test_turnover_zscore_requires_full_reference_window() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 10.0, 10.0, 10.0, 10.0],
            "volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = turnover_zscore(frame, length=4)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].notna().all()


def test_turnover_zscore_leaves_zero_variance_reference_missing() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 10.0, 10.0, 10.0],
            "volume": [100.0, 100.0, 100.0, 200.0],
        }
    )

    result = turnover_zscore(frame, length=4)

    assert result.isna().all()


def test_turnover_zscore_allows_non_temporal_index_and_preserves_order() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 10.0, 10.0, 10.0],
            "volume": [10.0, 20.0, 30.0, 40.0],
        },
        index=pd.Index([2, 0, 3, 1]),
    )

    result = turnover_zscore(frame, length=3)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_turnover_zscore_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = turnover_zscore(prices, length=3)

    pd.testing.assert_index_equal(result.index, prices.index)

    aaa = result.loc[("yfinance", "AAA.ST")]
    bbb = result.loc[("yfinance", "BBB.ST")]

    assert aaa.iloc[:2].isna().all()
    assert aaa.iloc[2:].notna().all()

    # BBB.ST has constant turnover. It must remain missing after warm-up rather
    # than using turnover observations from AAA.ST at the ticker boundary.
    assert bbb.isna().all()


def test_turnover_zscore_requires_adjusted_close_and_volume() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_close": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns: volume"):
        turnover_zscore(frame, length=2)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_turnover_zscore_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        turnover_zscore(
            pd.DataFrame(
                {
                    "adjusted_close": [10.0, 11.0],
                    "volume": [100, 200],
                }
            ),
            length=length,  # type: ignore[arg-type]
        )


def test_turnover_zscore_rejects_length_below_two() -> None:
    with pytest.raises(ValueError, match="must be at least 2"):
        turnover_zscore(
            pd.DataFrame(
                {
                    "adjusted_close": [10.0],
                    "volume": [100],
                }
            ),
            length=1,
        )


@pytest.mark.parametrize("log", [0, 1, "true", None])
def test_turnover_zscore_rejects_non_boolean_log(log: object) -> None:
    with pytest.raises(ValueError, match="log parameter must be a boolean"):
        turnover_zscore(
            pd.DataFrame(
                {
                    "adjusted_close": [10.0, 11.0],
                    "volume": [100, 200],
                }
            ),
            length=2,
            log=log,  # type: ignore[arg-type]
        )


def test_turnover_zscore_validates_parameters_before_columns() -> None:
    frame = pd.DataFrame({"wrong_column": [1.0]})

    with pytest.raises(ValueError, match="must be at least 2"):
        turnover_zscore(frame, length=1)

    with pytest.raises(ValueError, match="log parameter must be a boolean"):
        turnover_zscore(
            frame,
            length=2,
            log=1,  # type: ignore[arg-type]
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
            "volume": [1000, 1100, 1200, 1300, 500, 500, 500, 500],
        }
    )
