"""Feature transformations for historical adjusted-close returns."""

import pandas as pd

from swingtrader.data.features._numerical import safe_divide
from swingtrader.data.features._validation import validate_feature_input


def add_return_features(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Add trailing percentage-return columns for each requested horizon.

    The input must contain provider, ticker, and trading_date identifiers either
    as columns or named index levels, plus an adjusted_close column. Returns are
    calculated independently within each provider/ticker group and the input row
    order is preserved.
    """
    _validate_horizons(horizons)

    validate_feature_input(
        prices,
        required_columns={"adjusted_close"},
    )

    data = prices.copy()
    by_ticker = data.groupby(["provider", "ticker"], sort=False)

    for horizon in horizons:
        previous_price = by_ticker["adjusted_close"].shift(horizon)

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
