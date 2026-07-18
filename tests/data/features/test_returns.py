import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features.returns import add_return_features


def test_add_return_features_preserves_source_columns_and_adds_returns() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = add_return_features(prices, horizons=(1, 2))

    assert list(result.columns) == [*prices.columns, "return_1d", "return_2d"]
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)
    pd.testing.assert_series_equal(
        result["return_1d"],
        pd.Series([np.nan, 0.1, 0.1, np.nan, -0.2, 0.1], index=prices.index, name="return_1d"),
    )
    pd.testing.assert_series_equal(
        result["return_2d"],
        pd.Series(
            [np.nan, np.nan, 0.21, np.nan, np.nan, -0.12], index=prices.index, name="return_2d"
        ),
    )


def test_add_return_features_calculates_each_ticker_independently() -> None:
    prices = _prices()

    result = add_return_features(prices, horizons=(1,))

    # The first BBB.ST row is a warm-up NaN; if AAA.ST's tail leaked across the
    # ticker boundary this would instead be a finite return.
    assert np.isnan(result.loc[("yfinance", "BBB.ST"), "return_1d"].iloc[0])
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST"), "return_1d"].reset_index(drop=True),
        pd.Series([np.nan, 0.1, 0.1], name="return_1d"),
    )


def test_add_return_features_rejects_identifiers_as_columns() -> None:
    prices = _prices().reset_index()

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        add_return_features(prices, horizons=(1,))


def test_add_return_features_rejects_unsorted_index() -> None:
    prices = _prices().iloc[[1, 0, 2, 3, 4, 5]]

    with pytest.raises(ValueError, match="must be sorted"):
        add_return_features(prices, horizons=(1,))


@pytest.mark.parametrize(
    "horizons",
    [(), (1, 1), (0,), (-1,), (True,), (1.5,)],
)
def test_add_return_features_rejects_invalid_horizons(horizons: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_return_features(_prices().iloc[:1], horizons=horizons)


def test_add_return_features_runs_dataframe_validation() -> None:
    with pytest.raises(ValueError, match="Missing required columns: adjusted_close"):
        add_return_features(_prices().drop(columns="adjusted_close"), horizons=(1,))


def test_add_return_features_handles_empty_input() -> None:
    prices = pd.DataFrame(
        {"adjusted_close": pd.Series(dtype="float64")},
        index=pd.MultiIndex.from_arrays([[], [], []], names=["provider", "ticker", "trading_date"]),
    )

    result = add_return_features(prices, horizons=(1, 5))

    assert result.empty
    assert result["return_1d"].dtype == "float64"
    assert result["return_5d"].dtype == "float64"


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
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
            ),
            "adjusted_close": [100.0, 110.0, 121.0, 50.0, 40.0, 44.0],
        }
    ).set_index(["provider", "ticker", "trading_date"])
