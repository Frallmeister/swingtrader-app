import pandas as pd
import pytest

from swingtrader.indicators._smoothing import wilder_moving_average
from swingtrader.indicators.volatility import (
    atr,
    atr_percent,
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_percent_b,
    true_range,
)


def test_true_range_returns_expected_values() -> None:
    frame = _prices().iloc[:4]

    result = true_range(frame)

    assert isinstance(result, pd.Series)
    assert result.name == "true_range"
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_series_equal(
        result,
        pd.Series([2.0, 3.0, 3.0, 1.0], name="true_range"),
        check_exact=False,
    )


def test_true_range_first_row_falls_back_to_high_low() -> None:
    frame = _prices().iloc[:4]

    result = true_range(frame)

    assert result.iloc[0] == frame["high"].iloc[0] - frame["low"].iloc[0]


def test_true_range_allows_non_temporal_index_and_preserves_row_order() -> None:
    frame = _prices().iloc[:3].set_axis(pd.Index([2, 0, 1]))

    result = true_range(frame)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_true_range_requires_high_low_close() -> None:
    frame = _prices().iloc[:4].drop(columns="close")

    with pytest.raises(ValueError, match="Missing required columns"):
        true_range(frame)


def test_true_range_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = true_range(prices)

    pd.testing.assert_index_equal(result.index, prices.index)
    # A constant price yields a True Range of zero for every row, isolated from
    # AAA.ST.
    assert (result.loc[("yfinance", "BBB.ST")] == 0.0).all()
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        pd.Series([2.0, 3.0, 3.0, 1.0], name="true_range"),
        check_exact=False,
    )


def test_atr_returns_wilder_smoothed_true_range() -> None:
    frame = _prices().iloc[:4]

    result = atr(frame, length=2)

    assert result.name == "atr"
    expected = wilder_moving_average(true_range(frame), length=2).rename("atr")
    pd.testing.assert_series_equal(result, expected, check_exact=False)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_atr_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        atr(_prices().iloc[:4], length=length)  # type: ignore[arg-type]


def test_atr_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = atr(prices, length=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    bbb_atr = result.loc[("yfinance", "BBB.ST")]
    assert (bbb_atr.dropna() == 0.0).all()
    assert bbb_atr.isna().sum() == 1


def test_atr_percent_scales_atr_by_close() -> None:
    frame = _prices().iloc[:4]

    result = atr_percent(frame, length=2)

    assert result.name == "atr_percent"
    expected = (100 * atr(frame, length=2) / frame["close"]).rename("atr_percent")
    pd.testing.assert_series_equal(result, expected, check_exact=False)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_atr_percent_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        atr_percent(_prices().iloc[:4], length=length)  # type: ignore[arg-type]


def test_atr_percent_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = atr_percent(prices, length=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    bbb_atr_percent = result.loc[("yfinance", "BBB.ST")]
    assert (bbb_atr_percent.dropna() == 0.0).all()
    assert bbb_atr_percent.isna().sum() == 1


def test_bollinger_bands_returns_expected_columns_and_values() -> None:
    close = _close()

    result = bollinger_bands(close, length=3, num_std=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["bollinger_middle", "bollinger_upper", "bollinger_lower"]
    pd.testing.assert_index_equal(result.index, close.index)

    rolling = close.rolling(window=3, min_periods=3)
    expected_middle = rolling.mean()
    # Population standard deviation (ddof=0), matching Bollinger's definition.
    expected_std = rolling.std(ddof=0)
    pd.testing.assert_series_equal(
        result["bollinger_middle"],
        expected_middle.rename("bollinger_middle"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["bollinger_upper"],
        (expected_middle + 2 * expected_std).rename("bollinger_upper"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["bollinger_lower"],
        (expected_middle - 2 * expected_std).rename("bollinger_lower"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["bollinger_middle"].dropna().reset_index(drop=True),
        pd.Series([12.0, 13.0], name="bollinger_middle"),
        check_exact=False,
    )


def test_bollinger_bands_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="close")

    result = bollinger_bands(values, length=2, num_std=2)

    pd.testing.assert_index_equal(result.index, values.index)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_bollinger_bands_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        bollinger_bands(_close(), length=length)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_std", [0, -1, True, "2"])
def test_bollinger_bands_rejects_invalid_num_std(num_std: object) -> None:
    with pytest.raises(ValueError, match="positive number"):
        bollinger_bands(_close(), length=3, num_std=num_std)  # type: ignore[arg-type]


def test_bollinger_bands_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = bollinger_bands(close, length=3, num_std=2)

    assert list(result.columns) == ["bollinger_middle", "bollinger_upper", "bollinger_lower"]
    pd.testing.assert_index_equal(result.index, close.index)
    # A constant BBB.ST price collapses the bands onto the middle band, isolated
    # from AAA.ST.
    bbb = result.loc[("yfinance", "BBB.ST")].dropna()
    assert (bbb["bollinger_upper"] == bbb["bollinger_middle"]).all()
    assert (bbb["bollinger_lower"] == bbb["bollinger_middle"]).all()
    assert (bbb["bollinger_middle"] == 100.0).all()


def test_bollinger_bandwidth_matches_band_width_over_middle() -> None:
    close = _close()

    result = bollinger_bandwidth(close, length=3, num_std=2)

    assert result.name == "bollinger_bandwidth"
    bands = bollinger_bands(close, length=3, num_std=2)
    expected = (
        (bands["bollinger_upper"] - bands["bollinger_lower"]) / bands["bollinger_middle"]
    ).rename("bollinger_bandwidth")
    pd.testing.assert_series_equal(result, expected, check_exact=False)


def test_bollinger_bandwidth_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = bollinger_bandwidth(close, length=3, num_std=2)

    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price has zero band width, so bandwidth is zero after warm-up.
    bbb = result.loc[("yfinance", "BBB.ST")]
    assert (bbb.dropna() == 0.0).all()
    assert bbb.notna().sum() == 2


@pytest.mark.parametrize("num_std", [0, -1, True, "2"])
def test_bollinger_bandwidth_rejects_invalid_num_std(num_std: object) -> None:
    with pytest.raises(ValueError, match="positive number"):
        bollinger_bandwidth(_close(), length=3, num_std=num_std)  # type: ignore[arg-type]


def test_bollinger_percent_b_locates_value_within_bands() -> None:
    close = _close()

    result = bollinger_percent_b(close, length=3, num_std=2)

    assert result.name == "bollinger_percent_b"
    bands = bollinger_bands(close, length=3, num_std=2)
    expected = (
        (close - bands["bollinger_lower"]) / (bands["bollinger_upper"] - bands["bollinger_lower"])
    ).rename("bollinger_percent_b")
    pd.testing.assert_series_equal(result, expected, check_exact=False)
    # A value equal to the middle band sits exactly halfway between the bands.
    assert result.iloc[3] == pytest.approx(0.5)


def test_bollinger_percent_b_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = bollinger_percent_b(close, length=3, num_std=2)

    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price has zero band width, so %B is undefined (NA).
    assert result.loc[("yfinance", "BBB.ST")].isna().all()
    assert result.loc[("yfinance", "AAA.ST")].notna().sum() == 2


@pytest.mark.parametrize("num_std", [0, -1, True, "2"])
def test_bollinger_percent_b_rejects_invalid_num_std(num_std: object) -> None:
    with pytest.raises(ValueError, match="positive number"):
        bollinger_percent_b(_close(), length=3, num_std=num_std)  # type: ignore[arg-type]


def _close() -> pd.Series:
    return _prices()["close"].iloc[:4]


def _multi_ticker_close() -> pd.Series:
    return _indexed_prices()["close"]


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
            "high": [11.0, 13.0, 15.0, 14.0, 100.0, 100.0, 100.0, 100.0],
            "low": [9.0, 10.0, 12.0, 13.0, 100.0, 100.0, 100.0, 100.0],
            "close": [10.0, 12.0, 14.0, 13.0, 100.0, 100.0, 100.0, 100.0],
            "adjusted_close": [9.0, 11.0, 13.0, 12.5, 100.0, 100.0, 100.0, 100.0],
        }
    )
