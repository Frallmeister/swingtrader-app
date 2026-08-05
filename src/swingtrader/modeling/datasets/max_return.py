"""Maximum favorable-excursion target builders for canonical modeling datasets."""

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import (
    MARKET_PRICE_INDEX_NAMES,
    validate_market_price_index,
    validate_required_columns,
)

MAX_RETURN_HORIZONS = (5, 10, 15)

# These are logical inputs to the target family manifest. The three market
# identifiers are supplied by the canonical index rather than ordinary columns.
MAX_RETURN_REQUIRED_PRICE_COLUMNS = (*MARKET_PRICE_INDEX_NAMES, "open", "high")


def future_max_return_target_columns(horizons: tuple[int, ...]) -> tuple[str, ...]:
    """Return the ordered output schema for the configured horizons."""
    return tuple(f"future_max_return_{horizon}d" for horizon in horizons)


def add_future_max_return_targets(
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Append the maximum favorable return over the next ``horizon`` sessions.

    Entry is assumed at the next session's open. For each horizon the target is
    the highest intraday high across the following ``horizon`` sessions divided
    by that entry open, minus one. All outputs depend on future prices and must
    never be used as features.

    ``prices`` must use the canonical, unique, sorted market-price MultiIndex
    with levels ``provider``, ``ticker``, and ``trading_date``. Targets are
    calculated independently within each provider/ticker series and the input
    index is preserved. A target is missing when the entry open or any high in
    the window is unavailable, non-finite, or non-positive, which also leaves the
    final ``horizon`` sessions of every series missing.

    Parameters
    ----------
    prices : pandas.DataFrame
        Canonical market-price frame containing ``open`` and ``high`` columns.
    horizons : tuple of int
        Positive, unique session horizons to build targets for.

    Returns
    -------
    pandas.DataFrame
        A copy of ``prices`` with the original index and row order preserved and
        one ``future_max_return_{horizon}d`` (float) column appended per horizon.

    Raises
    ------
    ValueError
        If the index is not the canonical market-price index, if the horizons are
        invalid, or if a required price column is missing.
    """
    validate_market_price_index(prices)
    _validate_horizons(horizons)
    validate_required_columns(prices, required_columns={"open", "high"})

    result = prices.copy()
    open_ = _finite_positive(result["open"])
    high = _finite_positive(result["high"])
    grouped_open = open_.groupby(level=["provider", "ticker"], sort=False)
    grouped_high = high.groupby(level=["provider", "ticker"], sort=False)
    entry_open = grouped_open.shift(-1)
    for horizon in horizons:
        future_highs = pd.concat(
            [grouped_high.shift(-step) for step in range(1, horizon + 1)],
            axis="columns",
        )
        max_future_high = future_highs.max(axis="columns", skipna=False)
        result[f"future_max_return_{horizon}d"] = (max_future_high / entry_open - 1).astype(
            "float64"
        )
    return result


def _finite_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    return numeric.mask(numeric.le(0) | ~np.isfinite(numeric))


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons:
        raise ValueError("At least one maximum-return target horizon is required.")
    if len(horizons) != len(set(horizons)):
        raise ValueError("Maximum-return target horizons must be unique.")
    if any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("Maximum-return target horizons must be positive integers.")
