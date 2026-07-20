"""Feature transformations for historical adjusted-close returns."""

import pandas as pd

from swingtrader.core.numerical import safe_divide
from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_required_columns,
)


def add_return_features(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Return a copy of prices with trailing percentage-return features added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus an
    ``adjusted_close`` column. The index must be unique and sorted. Returns are
    calculated independently within each provider/ticker group and the input row
    order is preserved.
    """
    validate_market_price_index(prices)
    validate_required_columns(prices, required_columns={"adjusted_close"})
    _validate_horizons(horizons)

    data = prices.copy()
    if data.empty:
        for horizon in horizons:
            data[f"return_{horizon}d"] = pd.Series(index=data.index, dtype="float64")
        return data

    adjusted_close_by_ticker = data.loc[:, "adjusted_close"].groupby(
        level=["provider", "ticker"],
        sort=False,
    )

    for horizon in horizons:
        previous_price = adjusted_close_by_ticker.shift(horizon)

        data[f"return_{horizon}d"] = safe_divide(
            data["adjusted_close"],
            previous_price,
        ).sub(1)

    return data


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons:
        raise ValueError("At least one return horizon is required.")

    if len(horizons) != len(set(horizons)):
        raise ValueError("Return horizons must be unique.")

    if any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("Return horizons must be positive integers.")
