"""Cross-sectional return-strength and market-context features."""

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)

_CROSS_SECTION_LEVELS = ["provider", "trading_date"]


def add_cross_sectional_features(
    data: pd.DataFrame,
    *,
    return_horizons: tuple[int, ...] = (1, 5, 10, 20),
    market_return_horizon: int = 1,
    minimum_cross_section_size: int = 2,
) -> pd.DataFrame:
    """Append same-date relative-strength and market-context features.

    Return percentiles compare stocks within each provider and trading date. Market
    breadth, equal-weight return, and median return use the selected trailing-return
    horizon and are repeated for every stock in the same cross-section.
    """
    validate_market_price_index(data)
    _validate_horizons(return_horizons)
    _validate_market_return_horizon(market_return_horizon)
    _validate_minimum_cross_section_size(minimum_cross_section_size)

    return_columns = {f"return_{horizon}d" for horizon in return_horizons}
    market_return_column = f"return_{market_return_horizon}d"
    validate_required_columns(
        data,
        required_columns={*return_columns, market_return_column},
    )

    percentile_columns = [
        f"return_{horizon}d_cross_sectional_percentile" for horizon in return_horizons
    ]
    market_columns = [
        f"market_breadth_positive_{market_return_horizon}d",
        f"market_equal_weight_return_{market_return_horizon}d",
        f"market_median_return_{market_return_horizon}d",
    ]
    validate_new_columns(data, new_columns=[*percentile_columns, *market_columns])

    result = data.copy()
    for horizon, output_column in zip(return_horizons, percentile_columns, strict=True):
        returns = _finite_float(result[f"return_{horizon}d"]).where(
            _uses_shared_provider_calendar(result.index, horizon=horizon, future=False)
        )
        result[output_column] = _cross_sectional_percentile(
            returns,
            minimum_cross_section_size=minimum_cross_section_size,
        )

    market_returns = _finite_float(result[market_return_column]).where(
        _uses_shared_provider_calendar(
            result.index,
            horizon=market_return_horizon,
            future=False,
        )
    )
    grouped_market_returns = market_returns.groupby(level=_CROSS_SECTION_LEVELS, sort=False)
    valid_cross_section = grouped_market_returns.transform("count").ge(minimum_cross_section_size)
    positive = market_returns.gt(0).where(market_returns.notna()).astype("float64")

    result[market_columns[0]] = (
        positive.groupby(
            level=_CROSS_SECTION_LEVELS,
            sort=False,
        )
        .transform("mean")
        .where(valid_cross_section)
    )
    result[market_columns[1]] = (
        grouped_market_returns.transform("mean").where(valid_cross_section).astype("float64")
    )
    result[market_columns[2]] = (
        grouped_market_returns.transform("median").where(valid_cross_section).astype("float64")
    )
    return result


def _cross_sectional_percentile(
    values: pd.Series,
    *,
    minimum_cross_section_size: int,
) -> pd.Series:
    grouped = values.groupby(level=_CROSS_SECTION_LEVELS, sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    percentiles = (ranks - 0.5) / counts
    return percentiles.where(counts.ge(minimum_cross_section_size)).astype("float64")


def _finite_float(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    return numeric.mask(~np.isfinite(numeric))


def _uses_shared_provider_calendar(
    index: pd.MultiIndex,
    *,
    horizon: int,
    future: bool,
) -> pd.Series:
    index_frame = index.to_frame(index=False)
    calendar = (
        index_frame.loc[:, ["provider", "trading_date"]]
        .drop_duplicates()
        .sort_values(["provider", "trading_date"])
    )
    periods = -horizon if future else horizon
    calendar["expected_date"] = calendar.groupby("provider", sort=False)["trading_date"].shift(
        periods
    )
    expected_by_date = calendar.set_index(["provider", "trading_date"])["expected_date"]
    row_dates = pd.MultiIndex.from_frame(index_frame.loc[:, ["provider", "trading_date"]])
    expected_date = pd.Series(
        expected_by_date.reindex(row_dates).to_numpy(),
        index=index,
    )
    observed_date = pd.Series(index_frame["trading_date"].to_numpy(), index=index)
    actual_date = observed_date.groupby(
        level=["provider", "ticker"],
        sort=False,
    ).shift(periods)
    return actual_date.eq(expected_date) & expected_date.notna()


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons:
        raise ValueError("At least one cross-sectional return horizon is required.")
    if len(horizons) != len(set(horizons)):
        raise ValueError("Cross-sectional return horizons must be unique.")
    if any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("Cross-sectional return horizons must be positive integers.")


def _validate_market_return_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Market return horizon must be a positive integer.")


def _validate_minimum_cross_section_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("Minimum cross-section size must be an integer of at least two.")
