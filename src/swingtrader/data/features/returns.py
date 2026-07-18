"""Feature transformations for historical adjusted-close returns."""

import pandas as pd

from swingtrader.data.features._numerical import safe_divide
from swingtrader.data.features._validation import validate_feature_input, validate_temporal_order


def add_return_features(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """Return a copy of prices with trailing percentage-return features added.

    The input must contain provider, ticker, and trading_date identifiers either
    as columns or named index levels, plus an adjusted_close column. Returns are
    calculated independently within each provider/ticker group and the input row
    order is preserved.
    """
    validate_feature_input(
        prices,
        required_columns={"adjusted_close"},
    )
    validate_temporal_order(prices)
    _validate_horizons(horizons)

    data = prices.copy()
    if data.empty:
        for horizon in horizons:
            data[f"return_{horizon}d"] = pd.Series(index=data.index, dtype="float64")
        return data

    adjusted_close_by_ticker = _grouped_series(data, data.loc[:, "adjusted_close"])

    for horizon in horizons:
        previous_price = adjusted_close_by_ticker.shift(horizon)

        data[f"return_{horizon}d"] = safe_divide(
            data["adjusted_close"],
            previous_price,
        ).sub(1)

    return data


def _grouped_series(data: pd.DataFrame, values: pd.Series) -> pd.core.groupby.SeriesGroupBy:
    identifiers = ("provider", "ticker")
    identifiers_set = set(identifiers)
    index_names = data.index.names
    columns = data.columns

    if identifiers_set.issubset(index_names):
        return values.groupby(
            [data.index.get_level_values(identifier) for identifier in identifiers],
            sort=False,
        )
    if identifiers_set.issubset(columns):
        return values.groupby(
            [data[identifier] for identifier in identifiers],
            sort=False,
        )
    raise ValueError("The identifiers 'provider' and 'ticker' must be in either index or columns")


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
