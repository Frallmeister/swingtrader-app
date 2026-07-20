import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features.trend import add_trend_features
from swingtrader.indicators.moving_averages import sma


def test_add_trend_features_preserves_source_columns_and_adds_final_features() -> None:
    prices = _indexed_prices()
    original = prices.copy(deep=True)

    result = add_trend_features(
        prices,
        ma_lengths=(1, 2, 3),
    )

    expected_columns = [
        *prices.columns,
        "ema_fast_to_ema_mid",
        "ema_mid_to_ema_slow",
        "ema_mid_to_sma_mid",
        "close_to_ema_fast",
        "close_to_ema_mid",
        "close_to_ema_slow",
        "adx",
        "plus_di",
        "minus_di",
    ]
    assert list(result.columns) == expected_columns
    assert "sma_mid" not in result.columns
    assert "ema_fast" not in result.columns
    assert "ema_mid" not in result.columns
    assert "ema_slow" not in result.columns
    assert "ppo" not in result.columns
    assert "ppo_signal" not in result.columns
    assert "ppo_histogram" not in result.columns
    assert "ppo_percentile" not in result.columns
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    pd.testing.assert_series_equal(
        result["close_to_ema_slow"].reset_index(drop=True),
        pd.Series(
            [np.nan, np.nan, 14.0 / 12.5 - 1.0, 16.0 / 14.25 - 1.0, np.nan, np.nan, 0.0, 0.0],
            name="close_to_ema_slow",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ema_mid_to_sma_mid"].reset_index(drop=True),
        pd.Series(
            [
                np.nan,
                11.333333333333332 / 11.0 - 1.0,
                13.11111111111111 / 13.0 - 1.0,
                15.037037037037036 / 15.0 - 1.0,
                np.nan,
                0.0,
                0.0,
                0.0,
            ],
            name="ema_mid_to_sma_mid",
        ),
        check_exact=False,
    )


def test_add_trend_features_preserves_multiindex_and_calculates_each_ticker() -> None:
    prices = _indexed_prices()

    result = add_trend_features(
        prices,
        ma_lengths=(1, 2, 3),
    )

    pd.testing.assert_index_equal(result.index, prices.index)
    assert result.loc[("yfinance", "AAA.ST"), "close_to_ema_slow"].notna().sum() == 2
    assert result.loc[("yfinance", "BBB.ST"), "close_to_ema_slow"].notna().sum() == 2


def test_add_trend_features_rejects_identifiers_as_columns() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        add_trend_features(prices, ma_lengths=(1, 2, 3))


def test_add_trend_features_rejects_unsorted_input() -> None:
    prices = _indexed_prices().iloc[[1, 0, 2, 3, 4, 5, 6, 7]]

    with pytest.raises(ValueError, match="must be sorted"):
        add_trend_features(prices, ma_lengths=(1, 2, 3))


def test_trend_helpers_reject_invalid_inputs() -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="ascending order"):
        add_trend_features(prices, ma_lengths=(3, 2, 1))

    with pytest.raises(ValueError, match="positive integer"):
        sma(prices["adjusted_close"], length=0)

    with pytest.raises(ValueError, match="Missing required columns"):
        add_trend_features(prices.drop(columns="adjusted_close"))


def test_add_trend_features_uses_custom_adx_length() -> None:
    prices = _indexed_prices()

    default_length = add_trend_features(prices)
    custom_length = add_trend_features(prices, adx_length=2)

    # The default 14-row window never warms up on this short history, while the
    # short custom window produces populated directional-movement values.
    assert default_length["plus_di"].notna().sum() == 0
    assert custom_length["plus_di"].notna().sum() > 0


def test_add_trend_features_requires_high_low_close() -> None:
    prices = _indexed_prices().drop(columns="high")

    with pytest.raises(ValueError, match="Missing required columns"):
        add_trend_features(prices, ma_lengths=(1, 2, 3))


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
