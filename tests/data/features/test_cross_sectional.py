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
        minimum_cross_section_size=2,
    )

    evaluated = result.xs(_EVAL_DATE, level="trading_date")
    yfinance = evaluated.xs("yfinance", level="provider")
    assert yfinance["return_1d_cross_sectional_percentile"].tolist()[:4] == pytest.approx(
        [0.125, 0.375, 0.625, 0.875]
    )
    assert np.isnan(yfinance["return_1d_cross_sectional_percentile"].iloc[4])
    assert yfinance["market_breadth_positive_1d"].drop_duplicates().tolist() == [0.5]
    assert yfinance["market_mean_return_1d"].drop_duplicates().tolist() == pytest.approx(
        [0.005]
    )
    assert yfinance["market_median_return_1d"].drop_duplicates().tolist() == pytest.approx([0.005])

    other = evaluated.xs("other", level="provider")
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

    result = add_cross_sectional_features(
        data, return_horizons=(1,), market_return_horizon=1, minimum_cross_section_size=2
    )

    evaluated = result.xs(_EVAL_DATE, level="trading_date")
    assert evaluated["return_1d_cross_sectional_percentile"].tolist() == pytest.approx(
        [0.2, 0.2, 0.5, 0.7, 0.9]
    )


def test_add_cross_sectional_features_leaves_single_stock_cross_section_missing() -> None:
    result = add_cross_sectional_features(
        _return_frame([("yfinance", "AAA", 0.1)]),
        return_horizons=(1,),
        market_return_horizon=1,
    )

    assert result["return_1d_cross_sectional_percentile"].isna().all()
    assert result["market_breadth_positive_1d"].isna().all()
    assert result["market_mean_return_1d"].isna().all()
    assert result["market_median_return_1d"].isna().all()


def test_add_cross_sectional_features_requires_shared_provider_calendar_window() -> None:
    rows = [
        ("yfinance", "AAA", "2026-07-01", 100.0),
        ("yfinance", "AAA", "2026-07-02", 100.0),
        ("yfinance", "AAA", "2026-07-03", 102.0),
        ("yfinance", "BBB", "2026-07-01", 100.0),
        ("yfinance", "BBB", "2026-07-03", 130.0),
        ("yfinance", "CCC", "2026-07-01", 100.0),
        ("yfinance", "CCC", "2026-07-02", 100.0),
        ("yfinance", "CCC", "2026-07-03", 104.0),
    ]
    data = (
        pd.DataFrame(
            rows,
            columns=["provider", "ticker", "trading_date", "adjusted_close"],
        )
        .assign(trading_date=lambda frame: pd.to_datetime(frame["trading_date"]))
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )

    result = add_cross_sectional_features(
        data,
        return_horizons=(1,),
        market_return_horizon=1,
        minimum_cross_section_size=2,
    )
    final_date = result.xs(
        pd.Timestamp("2026-07-03"),
        level="trading_date",
    )

    assert final_date.loc[
        ("yfinance", "AAA"),
        "return_1d_cross_sectional_percentile",
    ] == pytest.approx(0.25)
    assert pd.isna(final_date.loc[("yfinance", "BBB"), "return_1d_cross_sectional_percentile"])
    assert final_date.loc[
        ("yfinance", "CCC"),
        "return_1d_cross_sectional_percentile",
    ] == pytest.approx(0.75)
    assert final_date["market_mean_return_1d"].dropna().unique().tolist() == (
        pytest.approx([0.03])
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


@pytest.mark.parametrize("size", [1, 0, -1, True, 1.5])
def test_add_cross_sectional_features_rejects_invalid_minimum_cross_section_size(
    size: int,
) -> None:
    with pytest.raises(ValueError, match="Minimum cross-section size"):
        add_cross_sectional_features(
            _return_frame([("yfinance", "AAA", 0.1)]),
            return_horizons=(1,),
            market_return_horizon=1,
            minimum_cross_section_size=size,
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


def test_add_cross_sectional_features_handles_empty_input() -> None:
    index = pd.MultiIndex.from_arrays(
        [[], [], []],
        names=["provider", "ticker", "trading_date"],
    )
    data = pd.DataFrame({"adjusted_close": pd.Series(index=index, dtype="float64")}, index=index)

    result = add_cross_sectional_features(data, return_horizons=(1,), market_return_horizon=1)

    assert result.empty
    assert all(pd.api.types.is_float_dtype(result[column]) for column in result.columns)


_WARMUP_DATE = pd.Timestamp("2026-06-30")
_EVAL_DATE = pd.Timestamp("2026-07-01")
_BASE_PRICE = 100.0


def _return_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    records = [
        (
            provider,
            ticker,
            date,
            _BASE_PRICE * (1 + value) if date == _EVAL_DATE else _BASE_PRICE,
        )
        for provider, ticker, value in rows
        for date in (_WARMUP_DATE, _EVAL_DATE)
    ]
    return (
        pd.DataFrame(
            records,
            columns=["provider", "ticker", "trading_date", "adjusted_close"],
        )
        .set_index(["provider", "ticker", "trading_date"])
        .sort_index()
    )
