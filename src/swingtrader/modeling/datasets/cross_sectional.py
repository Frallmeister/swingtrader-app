"""Cross-sectional targets derived from future returns."""

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_new_columns,
    validate_required_columns,
)
from swingtrader.modeling.datasets.labels import add_forward_return_targets

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
    relevance_grade_count: int = 16,
    minimum_cross_section_size: int = 20,
) -> pd.DataFrame:
    """Append market-relative, percentile, and ordinal future-return targets.

    Cross-sections are formed independently for each ``provider`` and ``trading_date``.
    Within each, the equal-weight mean future gross return (including the stock itself)
    serves as the benchmark: the market-relative target is the stock's gross return
    divided by that benchmark, minus one. Future returns are also ranked into a midpoint
    percentile, and the ordinal grade buckets that percentile into
    ``relevance_grade_count`` ordered levels.

    All outputs depend on future prices and must never be used as features.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical market-price frame indexed by a unique, sorted ``MultiIndex`` with
        levels ``provider``, ``ticker``, and ``trading_date`` and containing an
        ``adjusted_close`` column. Forward returns for each horizon are derived from
        ``adjusted_close`` internally, so no precomputed return columns are required.
    horizons : tuple of int
        Positive, unique forward-return horizons to build targets for.
    relevance_grade_count : int, optional
        Number of ordered relevance buckets, between two and 128. Grades range from
        ``0`` for the weakest future-return region to ``relevance_grade_count - 1`` for
        the strongest. Defaults to 16.
    minimum_cross_section_size : int, optional
        Smallest number of valid stocks a cross-section must contain before its targets
        are emitted; smaller groups stay missing. Must be at least two. Defaults to 20.

    Returns
    -------
    pandas.DataFrame
        A copy of ``data`` with the original index and row order preserved and, for each
        horizon, the following columns appended:

        - ``market_relative_forward_return_{horizon}d`` (float);
        - ``forward_return_{horizon}d_cross_sectional_percentile`` (float);
        - ``forward_return_{horizon}d_relevance_grade`` (nullable ``Int8``).

    Raises
    ------
    ValueError
        If the index is not the canonical market-price index, if the horizons,
        relevance grade count, or minimum cross-section size are invalid, if the
        ``adjusted_close`` column is missing, or if any output column already exists on
        ``data``.

    Notes
    -----
    Missing or non-finite forward returns are excluded from the benchmark and ranking.
    A forward return whose endpoint skipped one of the provider's trading sessions is
    excluded from that horizon's cross-section so every comparison spans the provider's
    shared calendar. The market-relative target is also left missing when the stock or
    benchmark gross return is non-positive. The canonical index has no market or universe
    identifier, so each provider/date group is treated as one comparable universe.
    """
    validate_market_price_index(data)
    _validate_horizons(horizons)
    _validate_relevance_grade_count(relevance_grade_count)
    _validate_minimum_cross_section_size(minimum_cross_section_size)

    output_columns = cross_sectional_return_target_columns(horizons)
    validate_required_columns(data, required_columns={"adjusted_close"})
    validate_new_columns(data, new_columns=output_columns)

    forward_return_frame = add_forward_return_targets(data, horizons=horizons)

    result = data.copy()
    for horizon in horizons:
        forward_returns = _finite_float(forward_return_frame[f"forward_return_{horizon}d"]).where(
            _uses_shared_provider_calendar(result.index, horizon=horizon, future=True)
        )
        grouped = forward_returns.groupby(level=_CROSS_SECTION_LEVELS, sort=False)
        valid_cross_section = grouped.transform("count").ge(minimum_cross_section_size)
        market_return = grouped.transform("mean").astype("float64")
        market_gross_return = market_return.add(1)
        stock_gross_return = forward_returns.add(1)
        valid_relative = stock_gross_return.gt(0) & market_gross_return.gt(0) & valid_cross_section

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
    """Rank ``values`` into midpoint percentiles within each provider/date group.

    Ties share an average rank, and the percentile is ``(rank - 0.5) / count`` so the
    result never reaches exactly zero or one. Groups with fewer than
    ``minimum_cross_section_size`` valid (non-missing) values yield all-missing
    percentiles.
    """
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
    """Flag rows whose ``horizon``-session return window aligns with the provider calendar.

    The per-provider calendar of distinct trading dates gives the expected endpoint that
    is ``horizon`` sessions away (backward when ``future`` is false, forward otherwise).
    A row is valid only when its own ``horizon``-shifted trading date matches that
    expected endpoint, which excludes stocks that skipped a provider session. Assumes
    rows are sorted by ``trading_date`` within each provider/ticker group, as guaranteed
    by the canonical index.
    """
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
        raise ValueError("Minimum cross-section size must be an integer of at least two.")
