import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators.moving_averages import ema, rolling_vwap, sma


def test_sma_and_ema_calculate_one_sequence() -> None:
    prices = _prices()["adjusted_close"].iloc[:4]

    simple = sma(prices, length=2)
    exponential = ema(prices, length=2)

    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 11.0, 13.0, 15.0], name="adjusted_close"),
    )
    pd.testing.assert_series_equal(
        exponential,
        pd.Series(
            [
                np.nan,
                11.333333333333332,
                13.11111111111111,
                15.037037037037036,
            ],
            name="adjusted_close",
        ),
        check_exact=False,
    )


def test_sma_and_ema_allow_ordered_datetime_index() -> None:
    values = pd.Series(
        [10.0, 12.0, 14.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
        name="adjusted_close",
    )

    simple = sma(values, length=2)
    exponential = ema(values, length=2)

    pd.testing.assert_index_equal(simple.index, values.index)
    pd.testing.assert_index_equal(exponential.index, values.index)
    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 11.0, 13.0], index=values.index, name="adjusted_close"),
    )


def test_sma_and_ema_reject_unordered_datetime_index() -> None:
    values = pd.Series(
        [10.0, 14.0, 12.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-03", "2026-07-02"]),
        name="adjusted_close",
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        sma(values, length=2)

    with pytest.raises(ValueError, match="chronologically ordered"):
        ema(values, length=2)


def test_sma_and_ema_allow_ordered_period_index() -> None:
    values = pd.Series(
        [10.0, 12.0, 14.0],
        index=pd.period_range("2026-07-01", periods=3, freq="D"),
        name="adjusted_close",
    )

    simple = sma(values, length=2)
    exponential = ema(values, length=2)

    pd.testing.assert_index_equal(simple.index, values.index)
    pd.testing.assert_index_equal(exponential.index, values.index)


def test_sma_and_ema_reject_unordered_trading_date_multiindex() -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance", "yfinance", "yfinance"],
            ["AAA.ST", "AAA.ST", "AAA.ST"],
            pd.to_datetime(["2026-07-01", "2026-07-03", "2026-07-02"]),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    values = pd.Series([10.0, 14.0, 12.0], index=index, name="adjusted_close")

    with pytest.raises(ValueError, match="must be sorted"):
        sma(values, length=2)

    with pytest.raises(ValueError, match="must be sorted"):
        ema(values, length=2)


def test_sma_and_ema_allow_non_temporal_index_and_preserve_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    simple = sma(values, length=2)
    exponential = ema(values, length=2)

    pd.testing.assert_index_equal(simple.index, values.index)
    pd.testing.assert_index_equal(exponential.index, values.index)
    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 12.0, 13.0], index=values.index, name="adjusted_close"),
    )


@pytest.mark.parametrize("length", [0, -1, True, 2.5, "2"])
def test_moving_averages_reject_invalid_lengths(length: object) -> None:
    values = _prices()["adjusted_close"]

    with pytest.raises(ValueError, match="positive integer"):
        sma(
            values,
            length=length,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="positive integer"):
        ema(
            values,
            length=length,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="positive integer"):
        rolling_vwap(
            _prices(),
            length=length,  # type: ignore[arg-type]
        )


def test_sma_groups_by_ticker_index_levels_without_cross_ticker_bleed() -> None:
    close = _multi_ticker_close()

    result = sma(close, length=2)

    pd.testing.assert_index_equal(result.index, close.index)
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        pd.Series([np.nan, 11.0, 13.0, 15.0], name="adjusted_close"),
    )
    # The first BBB.ST value is a warm-up NaN; if AAA.ST's tail bled across the
    # boundary this would instead be mean(16, 100) == 58.0.
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series([np.nan, 100.0, 100.0, 100.0], name="adjusted_close"),
    )


def test_ema_groups_by_ticker_index_levels_without_cross_ticker_bleed() -> None:
    close = _multi_ticker_close()

    result = ema(close, length=2)

    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price feeds an EMA of exactly that price after warm-up; any
    # cross-ticker state from AAA.ST would perturb these BBB.ST values.
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series([np.nan, 100.0, 100.0, 100.0], name="adjusted_close"),
    )


def test_primitives_reject_partial_multiindex() -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["AAA.ST", "AAA.ST", "AAA.ST"],
            pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
        ],
        names=["ticker", "trading_date"],
    )
    values = pd.Series([10.0, 12.0, 14.0], index=index, name="adjusted_close")

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        sma(values, length=2)


def test_primitives_reject_wrong_level_order() -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["AAA.ST", "AAA.ST", "AAA.ST"],
            ["yfinance", "yfinance", "yfinance"],
            pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
        ],
        names=["ticker", "provider", "trading_date"],
    )
    values = pd.Series([10.0, 12.0, 14.0], index=index, name="adjusted_close")

    with pytest.raises(ValueError, match="in that exact order"):
        sma(values, length=2)


@pytest.mark.parametrize(
    "indicator",
    [
        lambda values: sma(values, length=2),
        lambda values: ema(values, length=2),
    ],
)
def test_primitives_reject_unordered_dates_within_one_ticker(
    indicator: object,
) -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * 6,
            ["AAA.ST", "AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST", "BBB.ST"],
            pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-01",
                    "2026-07-03",
                    "2026-07-02",
                ]
            ),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    values = pd.Series([10.0, 11.0, 12.0, 100.0, 101.0, 102.0], index=index, name="adjusted_close")

    with pytest.raises(ValueError, match="must be sorted"):
        indicator(values)  # type: ignore[operator]


def test_rolling_vwap_uses_volume_weighted_typical_price() -> None:
    data = pd.DataFrame(
        {
            # Typical prices are 10, 13, and 15. They deliberately differ
            # from close so this test detects a close-only implementation.
            "high": [12.0, 18.0, 21.0],
            "low": [9.0, 9.0, 12.0],
            "close": [9.0, 12.0, 12.0],
            "volume": [1.0, 3.0, 2.0],
        }
    )

    result = rolling_vwap(data, length=2)

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            [np.nan, 12.25, 13.8],
            name="rolling_vwap",
        ),
    )


def test_rolling_vwap_groups_by_ticker_without_cross_ticker_bleed() -> None:
    prices = _prices().set_index(["provider", "ticker", "trading_date"])[
        ["high", "low", "close", "volume"]
    ]

    result = rolling_vwap(prices, length=2)

    pd.testing.assert_index_equal(result.index, prices.index)

    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        pd.Series(
            [np.nan, 11.5, 12.5, 15.5],
            name="rolling_vwap",
        ),
    )

    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series(
            [np.nan, 100.0, 100.0, 100.0],
            name="rolling_vwap",
        ),
    )


def test_rolling_vwap_returns_missing_for_zero_rolling_volume() -> None:
    data = pd.DataFrame(
        {
            "high": [11.0, 13.0, 15.0],
            "low": [9.0, 11.0, 13.0],
            "close": [10.0, 12.0, 14.0],
            "volume": [0.0, 0.0, 1.0],
        }
    )

    result = rolling_vwap(data, length=2)

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            [np.nan, np.nan, 14.0],
            name="rolling_vwap",
        ),
    )


@pytest.mark.parametrize("column", ["high", "low", "close", "volume"])
def test_rolling_vwap_requires_price_and_volume_columns(column: str) -> None:
    data = _prices().drop(columns=column)

    with pytest.raises(ValueError, match="Missing required columns"):
        rolling_vwap(data, length=2)


def _multi_ticker_close() -> pd.Series:
    return _prices().set_index(["provider", "ticker", "trading_date"])["adjusted_close"]


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
            "volume": [1.0, 3.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
