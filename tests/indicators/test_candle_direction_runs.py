import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators import candle_direction_runs


def test_candle_direction_runs_calculates_count_return_and_body_magnitude() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 10.0, 11.0, 13.0, 12.0, 10.0, 10.0],
            "high": [11.0, 12.0, 14.0, 14.0, 13.0, 11.0, 12.0],
            "low": [9.0, 9.0, 10.0, 11.0, 9.0, 9.0, 9.0],
            "close": [10.0, 11.0, 13.0, 12.0, 10.0, 10.0, 11.0],
        }
    )

    result = candle_direction_runs(data, atr_length=1)

    expected_run = pd.Series([0, 1, 2, -1, -2, 0, 1], dtype="Int64")
    expected_return = pd.Series(
        [
            0.0,
            0.1,
            0.3,
            12.0 / 13.0 - 1.0,
            10.0 / 13.0 - 1.0,
            0.0,
            0.1,
        ]
    )
    expected_body_atr = pd.Series(
        [0.0, 0.5, 7.0 / 6.0, -0.25, -11.0 / 12.0, 0.0, 0.5]
    )

    pd.testing.assert_series_equal(
        result["direction_run"], expected_run, check_names=False
    )
    pd.testing.assert_series_equal(
        result["direction_run_return"], expected_return, check_names=False
    )
    pd.testing.assert_series_equal(
        result["direction_run_body_atr"], expected_body_atr, check_names=False
    )


def test_candle_direction_runs_preserves_missing_values_and_resets_after_them() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, np.nan, 12.0],
            "high": [12.0, 13.0, 13.0, 14.0],
            "low": [9.0, 10.0, 10.0, 11.0],
            "close": [11.0, 12.0, 12.0, 13.0],
        }
    )

    result = candle_direction_runs(data, atr_length=1)

    expected_run = pd.Series([1, 2, pd.NA, 1], dtype="Int64")
    pd.testing.assert_series_equal(
        result["direction_run"], expected_run, check_names=False
    )
    assert result.loc[2].isna().all()
    assert result.loc[3, "direction_run_return"] == pytest.approx(1.0 / 12.0)


def test_candle_direction_runs_requires_complete_atr_history_for_active_run() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [12.0, 13.0, 14.0, 14.0],
            "low": [9.0, 10.0, 11.0, 11.0],
            "close": [11.0, 12.0, 13.0, 12.0],
        }
    )

    result = candle_direction_runs(data, atr_length=2)

    assert result["direction_run_body_atr"].iloc[:3].isna().all()
    assert result["direction_run_body_atr"].iloc[3] == pytest.approx(-1.0 / 3.0)


def test_candle_direction_runs_isolates_tickers_and_preserves_index() -> None:
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
            "open": [10.0, 10.0, 11.0, 20.0, 20.0, 21.0],
            "high": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
            "low": [9.0, 9.0, 10.0, 19.0, 19.0, 20.0],
            "close": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        },
        index=index,
    )

    result = candle_direction_runs(data, atr_length=1)

    pd.testing.assert_index_equal(result.index, data.index)
    for ticker in ("AAA.ST", "BBB.ST"):
        ticker_result = result.xs(ticker, level="ticker")
        assert ticker_result["direction_run"].tolist() == [0, 1, 2]


def test_candle_direction_runs_validates_inputs_and_does_not_mutate_data() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
        }
    )
    original = data.copy(deep=True)

    candle_direction_runs(data, atr_length=1)

    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ValueError, match="Length must be a positive integer"):
        candle_direction_runs(data, atr_length=0)
    with pytest.raises(ValueError, match="Missing required columns: open"):
        candle_direction_runs(data.drop(columns="open"))
