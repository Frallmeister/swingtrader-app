import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features.cross_sectional import add_cross_sectional_features


def test_add_cross_sectional_features_calculates_percentiles_and_market_context() -> None:
    data = _return_frame(
        [
            ("other", "AAA", -0.4),
            ("other", "BBB", 0.4),
            ("yfinance", "AAA", -0.02),
            ("yfinance", "BBB", 0.0),
            ("yfinance", "CCC", 0.01),
            ("yfinance", "DDD", 0.03),
            ("yfinance", "EEE", np.nan),
        ]
    )
    original = data.copy(deep=True)

    result = add_cross_sectional_features(
        data,
        return_horizons=(1,),
        market_return_horizon=1,
    )

    yfinance = result.xs("yfinance", level="provider")
    assert yfinance["return_1d_cross_sectional_percentile"].tolist()[:4] == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )
    assert np.isnan(yfinance["return_1d_cross_sectional_percentile"].iloc[4])
    assert yfinance["market_breadth_positive_1d"].drop_duplicates().tolist() == [0.5]
    assert yfinance["market_equal_weight_return_1d"].drop_duplicates().tolist() == pytest.approx(
        [0.005]
    )
    assert yfinance["market_median_return_1d"].drop_duplicates().tolist() == pytest.approx([0.005])

    other = result.xs("other", level="provider")
    assert other["return_1d_cross_sectional_percentile"].tolist() == pytest.approx([0.25, 0.75])
    assert other["market_breadth_positive_1d"].drop_duplicates().tolist() == [0.5]
    pd.testing.assert_frame_equal(data, original)


def test_add_cross_sectional_features_assigns_equal_percentiles_to_ties() -> None:
    data = _return_frame(
        [
            ("yfinance", "AAA", 0.1),
            ("yfinance", "BBB", 0.1),
            ("yfinance", "CCC", 0.2),
            ("yfinance", "DDD", 0.3),
            ("yfinance", "EEE", 0.4),
        ]
    )

    result = add_cross_sectional_features(data, return_horizons=(1,), market_return_horizon=1)

    assert result["return_1d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.2, 0.2, 0.5, 0.7, 0.9]
    )


@pytest.mark.parametrize("horizons", [(), (1, 1), (0,), (-1,), (True,), (1.5,)])
def test_add_cross_sectional_features_rejects_invalid_horizons(
    horizons: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_cross_sectional_features(
            _return_frame([("yfinance", "AAA", 0.1)]),
            return_horizons=horizons,
            market_return_horizon=1,
        )


@pytest.mark.parametrize("horizon", [0, -1, True, 1.5])
def test_add_cross_sectional_features_rejects_invalid_market_return_horizon(
    horizon: int,
) -> None:
    with pytest.raises(ValueError, match="Market return horizon"):
        add_cross_sectional_features(
            _return_frame([("yfinance", "AAA", 0.1)]),
            return_horizons=(1,),
            market_return_horizon=horizon,
        )


def test_add_cross_sectional_features_rejects_output_collisions() -> None:
    data = _return_frame([("yfinance", "AAA", 0.1)])
    data["market_breadth_positive_1d"] = 0.5

    with pytest.raises(ValueError, match="market_breadth_positive_1d"):
        add_cross_sectional_features(
            data,
            return_horizons=(1,),
            market_return_horizon=1,
        )


def test_add_cross_sectional_features_rejects_missing_return_column() -> None:
    data = _return_frame([("yfinance", "AAA", 0.1)])

    with pytest.raises(ValueError, match="return_5d"):
        add_cross_sectional_features(
            data,
            return_horizons=(1, 5),
            market_return_horizon=1,
        )


def test_add_cross_sectional_features_handles_empty_input() -> None:
    index = pd.MultiIndex.from_arrays(
        [[], [], []],
        names=["provider", "ticker", "trading_date"],
    )
    data = pd.DataFrame({"return_1d": pd.Series(index=index, dtype="float64")}, index=index)

    result = add_cross_sectional_features(data, return_horizons=(1,), market_return_horizon=1)

    assert result.empty
    assert all(pd.api.types.is_float_dtype(result[column]) for column in result.columns)


def _return_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "provider": [row[0] for row in rows],
                "ticker": [row[1] for row in rows],
                "trading_date": pd.Timestamp("2026-07-01"),
                "return_1d": [row[2] for row in rows],
            }
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )
