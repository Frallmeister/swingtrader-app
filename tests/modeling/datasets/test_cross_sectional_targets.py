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

    evaluated = result.xs(_EVAL_DATE, level="trading_date")
    assert evaluated["market_relative_forward_return_5d"].tolist() == pytest.approx(
        [0.9 / 1.1 - 1, 1.0 / 1.1 - 1, 0.0, 1.2 / 1.1 - 1, 1.3 / 1.1 - 1]
    )
    assert evaluated["forward_return_5d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    assert evaluated["forward_return_5d_relevance_grade"].tolist() == [0, 1, 2, 3, 4]
    assert str(evaluated["forward_return_5d_relevance_grade"].dtype) == "Int8"
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

    evaluated = result.xs(_EVAL_DATE, level="trading_date")
    assert evaluated["forward_return_5d_cross_sectional_percentile"].tolist()[:4] == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )
    assert pd.isna(evaluated["forward_return_5d_cross_sectional_percentile"].iloc[4])
    assert pd.isna(evaluated["forward_return_5d_relevance_grade"].iloc[4])
    assert evaluated["market_relative_forward_return_5d"].iloc[0] == pytest.approx(0.9 / 1.05 - 1)


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

    evaluated = result.xs(_EVAL_DATE, level="trading_date")
    other = evaluated.xs("other", level="provider")
    yfinance = evaluated.xs("yfinance", level="provider")
    assert other["forward_return_5d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.25, 0.75]
    )
    assert yfinance["forward_return_5d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.25, 0.75]
    )
    assert other["market_relative_forward_return_5d"].tolist() == pytest.approx([-0.4, 0.4])


def test_add_cross_sectional_return_targets_leave_single_stock_cross_section_missing() -> None:
    result = add_cross_sectional_return_targets(
        _forward_return_frame([("yfinance", "AAA", 0.1)]),
        horizons=(5,),
        relevance_grade_count=5,
    )

    assert result["market_relative_forward_return_5d"].isna().all()
    assert result["forward_return_5d_cross_sectional_percentile"].isna().all()
    assert result["forward_return_5d_relevance_grade"].isna().all()


def test_add_cross_sectional_return_targets_require_shared_provider_calendar_window() -> None:
    rows = [
        ("yfinance", "AAA", "2026-07-01", 0.10),
        ("yfinance", "AAA", "2026-07-02", 0.20),
        ("yfinance", "AAA", "2026-07-03", np.nan),
        ("yfinance", "BBB", "2026-07-01", 0.50),
        ("yfinance", "BBB", "2026-07-03", np.nan),
        ("yfinance", "CCC", "2026-07-01", 0.30),
        ("yfinance", "CCC", "2026-07-02", 0.40),
        ("yfinance", "CCC", "2026-07-03", np.nan),
    ]
    data = (
        pd.DataFrame(
            rows,
            columns=[
                "provider",
                "ticker",
                "trading_date",
                "forward_return_1d",
            ],
        )
        .assign(trading_date=lambda frame: pd.to_datetime(frame["trading_date"]))
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )

    result = add_cross_sectional_return_targets(
        data,
        horizons=(1,),
        relevance_grade_count=5,
    )
    first_date = result.xs(
        pd.Timestamp("2026-07-01"),
        level="trading_date",
    )

    assert first_date.loc[
        ("yfinance", "AAA"),
        "forward_return_1d_cross_sectional_percentile",
    ] == pytest.approx(0.25)
    assert pd.isna(
        first_date.loc[
            ("yfinance", "BBB"),
            "forward_return_1d_cross_sectional_percentile",
        ]
    )
    assert first_date.loc[
        ("yfinance", "CCC"),
        "forward_return_1d_cross_sectional_percentile",
    ] == pytest.approx(0.75)


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


@pytest.mark.parametrize("size", [1, 0, -1, True, 1.5])
def test_add_cross_sectional_return_targets_rejects_invalid_minimum_size(
    size: int,
) -> None:
    with pytest.raises(ValueError, match="Minimum cross-section size"):
        add_cross_sectional_return_targets(
            _forward_return_frame([("yfinance", "AAA", 0.1)]),
            horizons=(5,),
            relevance_grade_count=5,
            minimum_cross_section_size=size,
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
    assert pd.api.types.is_float_dtype(result["forward_return_5d_cross_sectional_percentile"])
    assert str(result["forward_return_5d_relevance_grade"].dtype) == "Int8"


_EVAL_DATE = pd.Timestamp("2026-07-01")
_CALENDAR = pd.date_range(_EVAL_DATE, periods=6, freq="B")


def _forward_return_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    records = [
        (provider, ticker, date, value if date == _EVAL_DATE else 0.0)
        for provider, ticker, value in rows
        for date in _CALENDAR
    ]
    return (
        pd.DataFrame(
            records,
            columns=["provider", "ticker", "trading_date", "forward_return_5d"],
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )
