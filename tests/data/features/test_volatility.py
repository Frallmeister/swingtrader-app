import pandas as pd
import pytest

from swingtrader.data.features._numerical import wilder_moving_average
from swingtrader.data.features.volatility import (
    add_volatility_features,
    atr,
    atr_percent,
    true_range,
)


def test_add_volatility_features_preserves_source_columns_and_adds_final_features() -> None:
    prices = _indexed_prices()
    original = prices.copy(deep=True)

    result = add_volatility_features(prices, atr_length=2)

    expected_columns = [*prices.columns, "atr_percent"]
    assert list(result.columns) == expected_columns
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    expected = atr_percent(prices, length=2).rename("atr_percent")
    pd.testing.assert_series_equal(result["atr_percent"], expected, check_exact=False)


def test_add_volatility_features_uses_custom_atr_length() -> None:
    prices = _indexed_prices()

    default_length = add_volatility_features(prices)
    custom_length = add_volatility_features(prices, atr_length=2)

    # The default 14-row window never warms up on this short history, while the
    # short custom window produces populated ATR-percent values.
    assert default_length["atr_percent"].notna().sum() == 0
    assert custom_length["atr_percent"].notna().sum() > 0


def test_add_volatility_features_preserves_multiindex_and_calculates_each_ticker() -> None:
    prices = _indexed_prices()

    result = add_volatility_features(prices, atr_length=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    # A constant BBB.ST price yields a zero ATR percent after warm-up, isolated
    # from AAA.ST.
    bbb_atr_percent = result.loc[("yfinance", "BBB.ST"), "atr_percent"]
    assert (bbb_atr_percent.dropna() == 0.0).all()
    assert result.loc[("yfinance", "AAA.ST"), "atr_percent"].notna().sum() == 3
    assert result.loc[("yfinance", "BBB.ST"), "atr_percent"].notna().sum() == 3


def test_add_volatility_features_rejects_identifiers_as_columns() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        add_volatility_features(prices, atr_length=2)


def test_add_volatility_features_rejects_unsorted_input() -> None:
    prices = _indexed_prices().iloc[[1, 0, 2, 3, 4, 5, 6, 7]]

    with pytest.raises(ValueError, match="must be sorted"):
        add_volatility_features(prices, atr_length=2)


def test_add_volatility_features_requires_high_low_close() -> None:
    prices = _indexed_prices().drop(columns="high")

    with pytest.raises(ValueError, match="Missing required columns"):
        add_volatility_features(prices)


def test_add_volatility_features_rejects_invalid_configuration() -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="positive integer"):
        add_volatility_features(prices, atr_length=0)


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
        }
    )
