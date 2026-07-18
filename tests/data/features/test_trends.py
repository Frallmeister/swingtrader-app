import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features.trends import (
    add_trend_features,
    ema,
    ppo,
    ppo_histogram,
    ppo_percentile,
    ppo_signal,
    sma,
)


def test_add_trend_features_calculates_grouped_features() -> None:
    prices = _prices()

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    expected_sma_ratio = pd.Series(
        [np.nan, np.nan, 13.0 / 12.0 - 1.0, 15.0 / 14.0 - 1.0, np.nan, np.nan, 0.0, 0.0],
        name="sma_fast_to_sma_slow",
    )
    pd.testing.assert_series_equal(
        result["sma_fast_to_sma_slow"],
        expected_sma_ratio,
        check_exact=False,
    )

    pd.testing.assert_series_equal(
        result["ppo"],
        result["ema_fast_to_ema_slow"].rename("ppo"),
        check_exact=False,
    )
    assert "ppo_signal" in result.columns
    assert "ppo_histogram" in result.columns
    pd.testing.assert_series_equal(
        result["ppo_percentile"],
        pd.Series(
            [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
            name="ppo_percentile",
        ),
        check_exact=False,
    )
    assert "ppo" not in prices.columns


def test_add_trend_features_accepts_identifiers_as_index_levels() -> None:
    prices = _prices().set_index(["provider", "ticker", "trading_date"])

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
    )

    pd.testing.assert_index_equal(result.index, prices.index)
    assert result.loc[("yfinance", "AAA.ST"), "sma_fast_to_sma_slow"].notna().sum() == 2
    assert result.loc[("yfinance", "BBB.ST"), "sma_fast_to_sma_slow"].notna().sum() == 2


def test_sma_and_ema_calculate_per_ticker_averages() -> None:
    prices = _prices()

    simple = sma(data=prices, length=2, source="adjusted_close")
    exponential = ema(data=prices, length=2, source="adjusted_close")

    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 11.0, 13.0, 15.0, np.nan, 100.0, 100.0, 100.0], name="adjusted_close"),
    )
    pd.testing.assert_series_equal(
        exponential,
        pd.Series(
            [
                np.nan,
                11.333333333333332,
                13.11111111111111,
                15.037037037037036,
                np.nan,
                100.0,
                100.0,
                100.0,
            ],
            name="adjusted_close",
        ),
        check_exact=False,
    )


def test_ppo_returns_percent_by_default_or_ratio_when_requested() -> None:
    prices = _prices().iloc[:4]

    ratio = ppo(prices, fast=2, slow=3, use_percent=False)
    percent = ppo(prices, fast=2, slow=3)

    expected_ratio = pd.Series(
        [np.nan, np.nan, 0.04888888888888887, 0.05523099415204676],
        name="adjusted_close",
    )
    pd.testing.assert_series_equal(ratio, expected_ratio, check_exact=False)
    pd.testing.assert_series_equal(percent, expected_ratio.mul(100), check_exact=False)


def test_ppo_signal_and_histogram_calculate_from_existing_ppo_columns() -> None:
    data = _prices().iloc[:4].assign(ppo=[0.0, 1.0, 2.0, 3.0])

    signal = ppo_signal(data, length=2)
    histogram = ppo_histogram(data.assign(ppo_signal=signal))

    expected_signal = pd.Series(
        [np.nan, 0.6666666666666666, 1.5555555555555556, 2.518518518518518],
        name="ppo",
    )
    expected_histogram = pd.Series([np.nan, 1.0 / 3.0, 4.0 / 9.0, 13.0 / 27.0])
    pd.testing.assert_series_equal(signal, expected_signal, check_exact=False)
    pd.testing.assert_series_equal(histogram, expected_histogram, check_exact=False)


def test_ppo_percentile_calculates_grouped_point_in_time_rank() -> None:
    data = _prices().assign(ppo=[1.0, 3.0, 2.0, 2.0, 5.0, 4.0, np.nan, 6.0])

    percentile = ppo_percentile(data, min_history=1)

    expected = pd.Series(
        [np.nan, 1.0, 0.5, 2.0 / 3.0, np.nan, 0.0, np.nan, 1.0],
        name="ppo",
    )
    pd.testing.assert_series_equal(percentile, expected, check_exact=False)


def test_trend_helpers_reject_invalid_inputs() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="fast length"):
        add_trend_features(prices, fast_slow_lengths=(3, 2), ppo_lengths=(2, 3, 2))

    with pytest.raises(ValueError, match="positive integer"):
        sma(data=prices, length=0, source="adjusted_close")

    with pytest.raises(ValueError, match="Source column"):
        ema(data=prices, length=2, source="missing")

    with pytest.raises(ValueError, match="fast length"):
        ppo(prices, fast=3, slow=2)

    with pytest.raises(ValueError, match="column named 'ppo'"):
        ppo_signal(prices)

    with pytest.raises(ValueError, match="Missing required columns: ppo"):
        ppo_percentile(prices)

    with pytest.raises(ValueError, match="positive integer"):
        ppo_percentile(prices.assign(ppo=0.0), min_history=0)

    with pytest.raises(ValueError, match="columns 'ppo' and 'ppo_signal'"):
        ppo_histogram(prices)


@pytest.mark.parametrize("length", [0, -1, True, 2.5, "2"])
def test_moving_averages_reject_invalid_lengths(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        sma(
            data=_prices(),
            length=length,  # type: ignore[arg-type]
            source="adjusted_close",
        )


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
            "adjusted_close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
