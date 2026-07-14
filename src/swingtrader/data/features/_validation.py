"""Shared dataframe validation for feature transformations."""

from collections.abc import Collection
from typing import Literal

import pandas as pd

_IDENTIFIERS = ("provider", "ticker", "trading_date")
_IDENTIFIER_SET = set(_IDENTIFIERS)


def validate_feature_input(
    data: pd.DataFrame,
    *,
    required_columns: Collection[str] = (),
) -> None:
    """Validate that feature input data has identifiers and required columns.

    Identifier fields must be provided consistently as columns or as named index
    levels. Additional feature-specific columns can be required by passing
    required_columns.
    """
    columns = set(data.columns)
    _resolve_identifier_location(data)

    missing_columns = set(required_columns).difference(columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")


def validate_temporal_order(data: pd.DataFrame) -> None:
    """Validate strictly increasing trading dates within each provider/ticker group."""
    if _resolve_identifier_location(data) == "columns":
        identifiers = data.loc[:, list(_IDENTIFIERS)]
    else:
        identifiers = data.index.to_frame(index=False).loc[:, list(_IDENTIFIERS)]

    trading_date = pd.to_datetime(identifiers["trading_date"])
    previous_date = trading_date.groupby(
        [identifiers["provider"], identifiers["ticker"]],
        sort=False,
    ).shift()

    if trading_date.le(previous_date).any():
        raise ValueError(
            "Observations must be strictly ordered by trading_date "
            "within each provider/ticker group."
        )


def _resolve_identifier_location(data: pd.DataFrame) -> Literal["columns", "index"]:
    columns = set(data.columns)
    index_names = {name for name in data.index.names if name is not None}

    column_identifiers = _IDENTIFIER_SET.intersection(columns)
    index_identifiers = _IDENTIFIER_SET.intersection(index_names)
    duplicate_identifiers = column_identifiers.intersection(index_identifiers)

    if duplicate_identifiers:
        duplicated = ", ".join(sorted(duplicate_identifiers))
        raise ValueError(
            f"Feature identifiers must not appear both as columns and index levels: {duplicated}."
        )

    if column_identifiers == _IDENTIFIER_SET:
        return "columns"

    if index_identifiers == _IDENTIFIER_SET:
        return "index"

    raise ValueError(
        "The identifiers 'provider', 'ticker', and 'trading_date' "
        "must all be columns or all be named index levels."
    )
