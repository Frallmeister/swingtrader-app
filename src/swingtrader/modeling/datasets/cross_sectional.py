"""Cross-sectional targets derived from future returns."""

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)

_CROSS_SECTION_LEVELS = ["provider", "trading_date"]


def cross_sectional_return_target_columns(horizons: tuple[int, ...]) -> tuple[str, ...]:
    """Return the ordered output schema for the configured horizons."""
    return tuple(
        column
        for horizon in horizons
        for column in (
            f"market_relative_forward_return_{horizon}d",
            f"forward_return_{horizon}d_cross_sectional_percentile",
            f"forward_return_{horizon}d_relevance_grade",
        )
    )


def add_cross_sectional_return_targets(
    data: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    relevance_grade_count: int,
    minimum_cross_section_size: int = 2,
) -> pd.DataFrame:
    """Append market-relative, percentile, and ordinal future-return targets.

    Cross-sections are formed independently for each provider and trading date.
    The market-relative return divides each stock's future gross return by the
    equal-weight mean future gross return. Percentiles use average ranks, and the
    ordinal grade maps those percentiles to ``0`` through
    ``relevance_grade_count - 1``.
    """
    validate_market_price_index(data)
    _validate_horizons(horizons)
    _validate_relevance_grade_count(relevance_grade_count)
    _validate_minimum_cross_section_size(minimum_cross_section_size)

    forward_return_columns = {f"forward_return_{horizon}d" for horizon in horizons}
    output_columns = cross_sectional_return_target_columns(horizons)
    validate_required_columns(data, required_columns=forward_return_columns)
    validate_new_columns(data, new_columns=output_columns)

    result = data.copy()
    for horizon in horizons:
        forward_returns = _finite_float(result[f"forward_return_{horizon}d"]).where(
            _uses_shared_provider_calendar(result.index, horizon=horizon, future=True)
        )
        grouped = forward_returns.groupby(level=_CROSS_SECTION_LEVELS, sort=False)
        valid_cross_section = grouped.transform("count").ge(
            minimum_cross_section_size
        )
        market_return = grouped.transform("mean").astype("float64")
        market_gross_return = market_return.add(1)
        stock_gross_return = forward_returns.add(1)
        valid_relative = (
            stock_gross_return.gt(0)
            & market_gross_return.gt(0)
            & valid_cross_section
        )

        relative_column = f"market_relative_forward_return_{horizon}d"
        relative_return = stock_gross_return.div(market_gross_return).sub(1)
        result[relative_column] = relative_return.where(valid_relative).astype("float64")

        percentile = _cross_sectional_percentile(
            forward_returns,
            minimum_cross_section_size=minimum_cross_section_size,
        )
        percentile_column = f"forward_return_{horizon}d_cross_sectional_percentile"
        result[percentile_column] = percentile

        relevance = pd.Series(pd.NA, index=result.index, dtype="Int8")
        valid_percentile = percentile.notna()
        relevance.loc[valid_percentile] = (
            np.floor(percentile.loc[valid_percentile] * relevance_grade_count)
            .clip(0, relevance_grade_count - 1)
            .astype("int8")
        )
        result[f"forward_return_{horizon}d_relevance_grade"] = relevance

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
    calendar["expected_date"] = calendar.groupby("provider", sort=False)[
        "trading_date"
    ].shift(periods)
    expected_by_date = calendar.set_index(["provider", "trading_date"])["expected_date"]
    row_dates = pd.MultiIndex.from_frame(
        index_frame.loc[:, ["provider", "trading_date"]]
    )
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
        raise ValueError("At least one cross-sectional target horizon is required.")
    if len(horizons) != len(set(horizons)):
        raise ValueError("Cross-sectional target horizons must be unique.")
    if any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("Cross-sectional target horizons must be positive integers.")


def _validate_relevance_grade_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= 128:
        raise ValueError("Relevance grade count must be an integer between two and 128.")


def _validate_minimum_cross_section_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError(
            "Minimum cross-section size must be an integer of at least two."
        )
