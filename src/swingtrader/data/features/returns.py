"""Feature transformations for historical adjusted-close returns.

Feature generators return dataframes containing only newly calculated feature
columns. Orchestrators return a copy of the input dataframe with those feature
columns appended.
"""

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
    _validate_horizons(horizons)

    validate_feature_input(
        prices,
        required_columns={"adjusted_close"},
    )
    validate_temporal_order(prices)

    data = prices.copy()
    features = return_features(data, horizons=horizons, run_validation=False)
    data[features.columns] = features

    return data


def return_features(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    *,
    source: str = "adjusted_close",
    run_validation: bool = True,
) -> pd.DataFrame:
    """Calculate trailing percentage-return feature columns.

    Returns a dataframe containing only newly calculated ``return_{horizon}d``
    columns. The output preserves the exact input index and row order.
    """
    _validate_horizons(horizons)

    if run_validation:
        validate_feature_input(prices, required_columns={source})
        validate_temporal_order(prices)

    features = pd.DataFrame(index=prices.index)

    if prices.empty:
        for horizon in horizons:
            features[f"return_{horizon}d"] = pd.Series(index=prices.index, dtype="float64")
        return features

    source_by_ticker = _grouped_source(prices, source)

    for horizon in horizons:
        previous_price = source_by_ticker.shift(horizon)

        features[f"return_{horizon}d"] = safe_divide(
            prices[source],
            previous_price,
        ).sub(1)

    return features


def _grouped_source(data: pd.DataFrame, source: str) -> pd.core.groupby.SeriesGroupBy:
    identifiers = ("provider", "ticker")
    identifiers_set = set(identifiers)
    index_names = data.index.names
    columns = data.columns
    values = data.loc[:, source]

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
