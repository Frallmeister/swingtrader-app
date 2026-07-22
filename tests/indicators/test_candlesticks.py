import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators import candle_geometry, candle_range_context


def test_candle_geometry_calculates_normalized_body_wicks_and_close_location() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 12.0, 11.0, 10.0],
            "high": [13.0, 13.0, 12.0, 10.0],
            "low": [9.0, 10.0, 10.0, 10.0],
            "close": [12.0, 11.0, 11.0, 10.0],
        }
    )

    result = candle_geometry(data)

    expected = pd.DataFrame(
        {
            "signed_body_fraction": [0.5, -1.0 / 3.0, 0.0, np.nan],
            "upper_wick_fraction": [0.25, 1.0 / 3.0, 0.5, np.nan],
            "lower_wick_fraction": [0.25, 1.0 / 3.0, 0.5, np.nan],
            "close_location": [0.75, 1.0 / 3.0, 0.5, np.nan],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_candle_geometry_does_not_infer_wicks_when_open_is_missing() -> None:
    data = pd.DataFrame(
        {
            "open": [np.nan],
            "high": [12.0],
            "low": [8.0],
            "close": [10.0],
        }
    )

    result = candle_geometry(data)

    open_dependent_columns = [
        "signed_body_fraction",
        "upper_wick_fraction",
        "lower_wick_fraction",
    ]
    assert result.loc[0, open_dependent_columns].isna().all()
    assert result.loc[0, "close_location"] == pytest.approx(0.5)


def test_candle_range_context_uses_atr_from_previous_row() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 13.0, 10.0, 15.0],
            "high": [12.0, 15.0, 13.0, 17.0],
            "low": [9.0, 12.0, 10.0, 14.0],
            "close": [11.0, 14.0, 12.0, 16.0],
        }
    )

    result = candle_range_context(data, atr_length=2, percentile_length=2)

    expected_range_atr = pd.Series([np.nan, np.nan, 4.0 / 3.5, 5.0 / 3.75])
    expected_gap_atr = pd.Series([np.nan, np.nan, -4.0 / 3.5, 3.0 / 3.75])
    pd.testing.assert_series_equal(
        result["range_atr"], expected_range_atr, check_names=False
    )
    pd.testing.assert_series_equal(result["gap_atr"], expected_gap_atr, check_names=False)


def test_candle_range_context_ranks_current_range_against_prior_ranges_only() -> None:
    ranges = pd.Series([2.0, 4.0, 3.0, 5.0])
    data = pd.DataFrame(
        {
            "open": 10.0,
            "high": 10.0 + ranges / 2.0,
            "low": 10.0 - ranges / 2.0,
            "close": 10.0,
        }
    )

    result = candle_range_context(data, atr_length=1, percentile_length=2)

    expected = pd.Series([np.nan, np.nan, 0.5, 1.0], name="range_percentile")
    pd.testing.assert_series_equal(result["range_percentile"], expected)


def test_candlestick_indicators_isolate_tickers_and_preserve_index() -> None:
    index = pd.MultiIndex.from_product(
        [
            ["yfinance"],
            ["AAA.ST", "BBB.ST"],
            pd.date_range("2026-01-01", periods=3, freq="D"),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "high": [11.0, 12.0, 13.0, 22.0, 24.0, 26.0],
            "low": [9.0, 8.0, 7.0, 18.0, 16.0, 14.0],
            "close": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        },
        index=index,
    )

    geometry = candle_geometry(data)
    context = candle_range_context(data, atr_length=1, percentile_length=2)

    pd.testing.assert_index_equal(geometry.index, data.index)
    pd.testing.assert_index_equal(context.index, data.index)
    for ticker in ("AAA.ST", "BBB.ST"):
        ticker_context = context.xs(ticker, level="ticker")
        assert ticker_context["range_percentile"].iloc[:2].isna().all()
        assert pd.isna(ticker_context["gap_atr"].iloc[0])


def test_candlestick_indicators_reject_unordered_datetime_index() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-01"]),
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        candle_geometry(data)


def test_candlestick_indicators_do_not_mutate_input() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
        }
    )
    original = data.copy(deep=True)

    candle_geometry(data)
    candle_range_context(data, atr_length=1, percentile_length=1)

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize("function", [candle_geometry, candle_range_context])
def test_candlestick_indicators_require_ohlc_columns(function) -> None:
    with pytest.raises(ValueError, match="Missing required columns: open"):
        function(pd.DataFrame({"high": [2.0], "low": [1.0], "close": [1.5]}))


@pytest.mark.parametrize(
    ("kwargs", "invalid_value"),
    [
        ({"atr_length": 0, "percentile_length": 20}, 0),
        ({"atr_length": 14, "percentile_length": 0}, 0),
        ({"atr_length": True, "percentile_length": 20}, True),
    ],
)
def test_candle_range_context_rejects_invalid_lengths(kwargs, invalid_value) -> None:
    data = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.0], "close": [1.0]}
    )

    with pytest.raises(ValueError, match=repr(invalid_value)):
        candle_range_context(data, **kwargs)
