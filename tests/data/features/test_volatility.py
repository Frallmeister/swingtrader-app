import pandas as pd
import pytest

from swingtrader.data.features.volatility import add_volatility_features
from swingtrader.indicators.volatility import (
    adr,
    atr_percent,
    bollinger_bandwidth,
    bollinger_percent_b,
)


def test_add_volatility_features_preserves_source_columns_and_adds_final_features() -> None:
    prices = _indexed_prices()
    original = prices.copy(deep=True)

    result = add_volatility_features(prices, adr_length=2, atr_length=2, bollinger_length=3)

    expected_columns = [
        *prices.columns,
        "adr_percent",
        "atr_percent",
        "bollinger_bandwidth",
        "bollinger_percent_b",
    ]
    assert list(result.columns) == expected_columns
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    adjusted_close = prices["adjusted_close"]
    adjustment_factor = adjusted_close.div(prices["close"])
    adjusted_hlc = prices[["high", "low", "close"]].mul(adjustment_factor, axis=0)
    adjusted_hlc["close"] = adjusted_close

    expected_adr = adr(adjusted_hlc, length=2)["adr_percent"]
    pd.testing.assert_series_equal(result["adr_percent"], expected_adr, check_exact=False)

    expected = atr_percent(adjusted_hlc, length=2).rename("atr_percent")
    pd.testing.assert_series_equal(result["atr_percent"], expected, check_exact=False)

    pd.testing.assert_series_equal(
        result["bollinger_bandwidth"],
        bollinger_bandwidth(adjusted_close, length=3, num_std=2.0).rename("bollinger_bandwidth"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["bollinger_percent_b"],
        bollinger_percent_b(adjusted_close, length=3, num_std=2.0).rename("bollinger_percent_b"),
        check_exact=False,
    )


def test_add_volatility_features_uses_custom_adr_length() -> None:
    prices = _indexed_prices()

    default_length = add_volatility_features(prices, atr_length=2)
    custom_length = add_volatility_features(
        prices,
        adr_length=2,
        atr_length=2,
    )

    # The default 20-row window never warms up on this short history, while the
    # short custom window produces populated ADR-percent values.
    assert default_length["adr_percent"].notna().sum() == 0
    assert custom_length["adr_percent"].notna().sum() > 0


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

    result = add_volatility_features(prices, adr_length=2, atr_length=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    # A constant BBB.ST price yields a zero ATR percent after warm-up, isolated from AAA.ST.
    bbb = result.loc[("yfinance", "BBB.ST")]
    assert (bbb["adr_percent"].dropna() == 0.0).all()
    assert (bbb["atr_percent"].dropna() == 0.0).all()
    assert result.loc[("yfinance", "AAA.ST"), "adr_percent"].notna().sum() == 3
    assert result.loc[("yfinance", "BBB.ST"), "adr_percent"].notna().sum() == 3
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


def test_add_volatility_features_requires_adjusted_close() -> None:
    prices = _indexed_prices().drop(columns="adjusted_close")

    with pytest.raises(ValueError, match="Missing required columns"):
        add_volatility_features(prices)


def test_add_volatility_features_rejects_invalid_configuration() -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="positive integer"):
        add_volatility_features(prices, adr_length=0)

    with pytest.raises(ValueError, match="positive integer"):
        add_volatility_features(prices, atr_length=0)

    with pytest.raises(ValueError, match="positive integer"):
        add_volatility_features(prices, bollinger_length=0)

    with pytest.raises(ValueError, match="positive number"):
        add_volatility_features(prices, bollinger_num_std=0)


def test_add_volatility_features_uses_custom_bollinger_length() -> None:
    prices = _indexed_prices()

    default_length = add_volatility_features(prices, atr_length=2)
    custom_length = add_volatility_features(prices, atr_length=2, bollinger_length=3)

    # The default 20-row window never warms up on this short history, while the
    # short custom window produces populated Bollinger bandwidth values.
    assert default_length["bollinger_bandwidth"].notna().sum() == 0
    assert custom_length["bollinger_bandwidth"].notna().sum() > 0


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
