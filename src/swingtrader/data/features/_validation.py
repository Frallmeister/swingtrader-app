"""
Shared validation for feature input dataframes.
"""

from collections.abc import Collection

import pandas as pd

_IDENTIFIERS = {"provider", "ticker", "trading_date"}


def validate_feature_input(
    data: pd.DataFrame,
    *,
    required_columns: Collection[str] = (),
) -> None:
    columns = set(data.columns)
    index_names = set(data.index.names)

    identifiers_are_columns = _IDENTIFIERS.issubset(columns)
    identifiers_are_index = _IDENTIFIERS.issubset(index_names)

    if identifiers_are_columns and identifiers_are_index:
        raise ValueError(
            "The identifiers 'provider', 'ticker', and 'trading_date' "
            "must not appear both as columns and index levels."
        )

    if not identifiers_are_columns and not identifiers_are_index:
        raise ValueError(
            "The identifiers 'provider', 'ticker', and 'trading_date' "
            "must all be columns or all be named index levels."
        )

    missing_columns = set(required_columns).difference(columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")
