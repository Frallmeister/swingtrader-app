import pandas as pd
import pytest

from swingtrader.indicators import rolling_level_interactions


def test_rolling_level_interactions_measure_acceptance_and_failed_breaks() -> None:
    data = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 13.0],
            "low": [8.0, 9.0, 10.0, 9.0, 10.0, 8.0, 7.0],
            "close": [9.0, 10.0, 11.0, 12.5, 13.0, 8.5, 8.5],
        }
    )

    result = rolling_level_interactions(data, length=3, atr_length=1)

    assert result.iloc[:3].isna().all().all()
    assert result.loc[3, "prior_high"] == pytest.approx(12.0)
    assert result.loc[3, "prior_low"] == pytest.approx(8.0)
    assert result.loc[3, "close_to_upper_atr"] == pytest.approx(0.25)
    assert result.loc[3, "breakout_high_strength"] == pytest.approx(0.25)

    assert result.loc[4, "close_to_upper_atr"] == pytest.approx(0.0)
    assert result.loc[4, "failed_break_high_strength"] == pytest.approx(0.25)
    assert result.loc[4, "breakout_high_strength"] == pytest.approx(0.0)

    assert result.loc[5, "close_to_lower_atr"] == pytest.approx(-0.125)
    assert result.loc[5, "breakout_low_strength"] == pytest.approx(0.125)

    assert result.loc[6, "close_to_lower_atr"] == pytest.approx(0.1)
    assert result.loc[6, "failed_break_low_strength"] == pytest.approx(0.2)
    assert result.loc[6, "breakout_low_strength"] == pytest.approx(0.0)


def test_rolling_level_interactions_exclude_current_row_from_levels() -> None:
    data = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0],
            "low": [8.0, 7.0, 6.0],
            "close": [9.0, 10.0, 11.5],
        }
    )

    result = rolling_level_interactions(data, length=2, atr_length=1)

    assert result.loc[2, "prior_high"] == pytest.approx(11.0)
    assert result.loc[2, "prior_low"] == pytest.approx(7.0)
    assert result.loc[2, "breakout_high_strength"] > 0.0


def test_rolling_level_interactions_isolate_tickers_and_preserve_index() -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    index = pd.MultiIndex.from_product(
        [["yfinance"], ["AAA.ST", "BBB.ST"], dates],
        names=["provider", "ticker", "trading_date"],
    )
    data = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 13.0, 20.0, 22.0, 24.0, 26.0],
            "low": [8.0, 9.0, 10.0, 11.0, 16.0, 18.0, 20.0, 22.0],
            "close": [9.0, 10.0, 11.0, 12.0, 18.0, 20.0, 22.0, 24.0],
        },
        index=index,
    )

    result = rolling_level_interactions(data, length=2, atr_length=1)

    pd.testing.assert_index_equal(result.index, data.index)
    aaa = result.xs("AAA.ST", level="ticker")
    bbb = result.xs("BBB.ST", level="ticker")
    normalized_columns = [
        "close_to_upper_atr",
        "close_to_lower_atr",
        "breakout_high_strength",
        "breakout_low_strength",
        "failed_break_high_strength",
        "failed_break_low_strength",
    ]
    pd.testing.assert_frame_equal(
        aaa[normalized_columns].reset_index(drop=True),
        bbb[normalized_columns].reset_index(drop=True),
    )
    assert aaa.iloc[:2].isna().all().all()


@pytest.mark.parametrize(("length", "atr_length"), [(0, 14), (20, 0), (True, 14)])
def test_rolling_level_interactions_reject_invalid_lengths(
    length: int,
    atr_length: int,
) -> None:
    data = pd.DataFrame({"high": [2.0], "low": [1.0], "close": [1.5]})

    with pytest.raises(ValueError):
        rolling_level_interactions(data, length=length, atr_length=atr_length)


def test_rolling_level_interactions_require_price_columns() -> None:
    data = pd.DataFrame({"high": [2.0], "low": [1.0]})

    with pytest.raises(ValueError, match="Missing required columns: close"):
        rolling_level_interactions(data)


def test_rolling_level_interactions_do_not_mutate_input() -> None:
    data = pd.DataFrame(
        {
            "high": [2.0, 3.0],
            "low": [1.0, 2.0],
            "close": [1.5, 2.5],
        }
    )
    original = data.copy(deep=True)

    rolling_level_interactions(data, length=1, atr_length=1)

    pd.testing.assert_frame_equal(data, original)
