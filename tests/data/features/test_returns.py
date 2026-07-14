import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features.returns import add_return_features


def test_add_return_features_calculates_returns_per_ticker() -> None:
    prices = pd.DataFrame(
        {
            "provider": ["yfinance"] * 6,
            "ticker": ["AAA.ST", "AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST", "BBB.ST"],
            "trading_date": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                ]
            ).date,
            "adjusted_close": [100.0, 110.0, 121.0, 50.0, 40.0, 44.0],
        }
    )

    result = add_return_features(prices, horizons=(1, 2))

    pd.testing.assert_series_equal(
        result["return_1d"],
        pd.Series([np.nan, 0.1, 0.1, np.nan, -0.2, 0.1], name="return_1d"),
    )
    pd.testing.assert_series_equal(
        result["return_2d"],
        pd.Series([np.nan, np.nan, 0.21, np.nan, np.nan, -0.12], name="return_2d"),
    )


def test_add_return_features_accepts_identifiers_as_index_levels() -> None:
    prices = pd.DataFrame(
        {
            "provider": ["yfinance", "yfinance", "yfinance"],
            "ticker": ["AAA.ST", "AAA.ST", "AAA.ST"],
            "trading_date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]).date,
            "adjusted_close": [100.0, 125.0, 100.0],
        }
    ).set_index(["provider", "ticker", "trading_date"])

    result = add_return_features(prices, horizons=(1,))

    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_series_equal(
        result["return_1d"],
        pd.Series([np.nan, 0.25, -0.2], index=prices.index, name="return_1d"),
    )


@pytest.mark.parametrize(
    "horizons",
    [(), (1, 1), (0,), (-1,), (True,), (1.5,)],
)
def test_add_return_features_rejects_invalid_horizons(horizons: tuple[int, ...]) -> None:
    prices = pd.DataFrame(
        {
            "provider": ["yfinance"],
            "ticker": ["AAA.ST"],
            "trading_date": [pd.Timestamp("2026-07-01").date()],
            "adjusted_close": [100.0],
        }
    )

    with pytest.raises(ValueError, match="horizon"):
        add_return_features(prices, horizons=horizons)


def test_add_return_features_handles_empty_input() -> None:
    prices = pd.DataFrame(columns=["provider", "ticker", "trading_date", "adjusted_close"])

    result = add_return_features(prices, horizons=(1, 5))

    assert result.empty
    assert result["return_1d"].dtype == "float64"
    assert result["return_5d"].dtype == "float64"
