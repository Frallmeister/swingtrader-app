import numpy as np
import pandas as pd
import pytest

from swingtrader.modeling.datasets.cross_sectional import add_cross_sectional_return_targets


def test_add_cross_sectional_return_targets_calculates_expected_outputs() -> None:
    data = _forward_return_frame(
        [
            ("yfinance", "AAA", -0.1),
            ("yfinance", "BBB", 0.0),
            ("yfinance", "CCC", 0.1),
            ("yfinance", "DDD", 0.2),
            ("yfinance", "EEE", 0.3),
        ]
    )
    original = data.copy(deep=True)

    result = add_cross_sectional_return_targets(
        data,
        horizons=(5,),
        relevance_grade_count=5,
    )

    assert result["market_relative_forward_return_5d"].tolist() == pytest.approx(
        [0.9 / 1.1 - 1, 1.0 / 1.1 - 1, 0.0, 1.2 / 1.1 - 1, 1.3 / 1.1 - 1]
    )
    assert result["forward_return_5d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    assert result["forward_return_5d_relevance_grade"].tolist() == [0, 1, 2, 3, 4]
    assert str(result["forward_return_5d_relevance_grade"].dtype) == "Int8"
    pd.testing.assert_frame_equal(data, original)


def test_add_cross_sectional_return_targets_excludes_missing_values() -> None:
    data = _forward_return_frame(
        [
            ("yfinance", "AAA", -0.1),
            ("yfinance", "BBB", 0.0),
            ("yfinance", "CCC", 0.1),
            ("yfinance", "DDD", 0.2),
            ("yfinance", "EEE", np.nan),
        ]
    )

    result = add_cross_sectional_return_targets(
        data,
        horizons=(5,),
        relevance_grade_count=5,
    )

    assert result["forward_return_5d_cross_sectional_percentile"].tolist()[:4] == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )
    assert pd.isna(result["forward_return_5d_cross_sectional_percentile"].iloc[4])
    assert pd.isna(result["forward_return_5d_relevance_grade"].iloc[4])
    assert result["market_relative_forward_return_5d"].iloc[0] == pytest.approx(
        0.9 / 1.05 - 1
    )


def test_add_cross_sectional_return_targets_separates_providers() -> None:
    data = _forward_return_frame(
        [
            ("other", "AAA", -0.4),
            ("other", "BBB", 0.4),
            ("yfinance", "AAA", 0.1),
            ("yfinance", "BBB", 0.2),
        ]
    )

    result = add_cross_sectional_return_targets(
        data,
        horizons=(5,),
        relevance_grade_count=5,
    )

    other = result.xs("other", level="provider")
    yfinance = result.xs("yfinance", level="provider")
    assert other["forward_return_5d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.25, 0.75]
    )
    assert yfinance["forward_return_5d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.25, 0.75]
    )
    assert other["market_relative_forward_return_5d"].tolist() == pytest.approx([-0.4, 0.4])


@pytest.mark.parametrize("horizons", [(), (5, 5), (0,), (-1,), (True,), (1.5,)])
def test_add_cross_sectional_return_targets_rejects_invalid_horizons(
    horizons: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_cross_sectional_return_targets(
            _forward_return_frame([("yfinance", "AAA", 0.1)]),
            horizons=horizons,
            relevance_grade_count=5,
        )


@pytest.mark.parametrize("count", [1, 0, -1, 129, True, 2.5])
def test_add_cross_sectional_return_targets_rejects_invalid_grade_count(count: int) -> None:
    with pytest.raises(ValueError, match="grade count"):
        add_cross_sectional_return_targets(
            _forward_return_frame([("yfinance", "AAA", 0.1)]),
            horizons=(5,),
            relevance_grade_count=count,
        )


def test_add_cross_sectional_return_targets_rejects_output_collisions() -> None:
    data = _forward_return_frame([("yfinance", "AAA", 0.1)])
    data["market_relative_forward_return_5d"] = 0.0

    with pytest.raises(ValueError, match="market_relative_forward_return_5d"):
        add_cross_sectional_return_targets(
            data,
            horizons=(5,),
            relevance_grade_count=5,
        )


def test_add_cross_sectional_return_targets_handles_empty_input() -> None:
    index = pd.MultiIndex.from_arrays(
        [[], [], []],
        names=["provider", "ticker", "trading_date"],
    )
    data = pd.DataFrame(
        {"forward_return_5d": pd.Series(index=index, dtype="float64")},
        index=index,
    )

    result = add_cross_sectional_return_targets(
        data,
        horizons=(5,),
        relevance_grade_count=5,
    )

    assert result.empty
    assert pd.api.types.is_float_dtype(result["market_relative_forward_return_5d"])
    assert pd.api.types.is_float_dtype(
        result["forward_return_5d_cross_sectional_percentile"]
    )
    assert str(result["forward_return_5d_relevance_grade"].dtype) == "Int8"


def _forward_return_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "provider": [row[0] for row in rows],
                "ticker": [row[1] for row in rows],
                "trading_date": pd.Timestamp("2026-07-01"),
                "forward_return_5d": [row[2] for row in rows],
            }
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )
