"""Canonical in-memory market-data contract shared by indicators and features.

This module defines and enforces the canonical market-price representation used
across the indicator and feature layers. It contains only pandas ``DataFrame``
and ``Series`` helpers; it does not introduce a custom dataframe wrapper class.

The canonical multi-instrument contract is a unique, sorted ``MultiIndex`` with
levels ``provider``, ``ticker``, and ``trading_date``, in that exact order, with
identifiers not also present as ordinary columns. Public indicators additionally
support a single-instrument ordered ``Series`` or ``DataFrame`` that only has to
be chronologically ordered.
"""

from collections.abc import Callable, Collection

import pandas as pd

MARKET_PRICE_INDEX_NAMES = ("provider", "ticker", "trading_date")


def validate_market_price_index(data: pd.DataFrame | pd.Series) -> None:
    """Validate the canonical market-price MultiIndex.

    Market-price data used by the feature layer must have a unique index sorted in
    canonical order with levels ``provider``, ``ticker``, and ``trading_date``, in
    that exact order. The identifiers must not also appear as ordinary columns.

    Raises
    ------
    ValueError
        Raised when the index is not the canonical MultiIndex, when it is not
        unique, when it is not sorted in canonical order, or when a DataFrame also
        carries an identifier as an ordinary column.
    """
    index = data.index
    if not isinstance(index, pd.MultiIndex) or tuple(index.names) != MARKET_PRICE_INDEX_NAMES:
        raise ValueError(
            "Market-price data must use a MultiIndex with levels "
            "'provider', 'ticker', and 'trading_date', in that exact order."
        )

    if isinstance(data, pd.DataFrame):
        duplicated = [name for name in MARKET_PRICE_INDEX_NAMES if name in data.columns]
        if duplicated:
            names = ", ".join(duplicated)
            raise ValueError(f"Market-price identifiers must not also appear as columns: {names}.")

    if not index.is_unique:
        raise ValueError(
            "Market-price data must have a unique index over "
            "'provider', 'ticker', and 'trading_date'."
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            "Market-price data must be sorted by 'provider', 'ticker', and "
            "'trading_date'. Call data.sort_index() before generating features."
        )


def validate_required_columns(
    data: pd.DataFrame,
    *,
    required_columns: Collection[str] = (),
) -> None:
    """Validate that market-frame input data contains the required value columns."""
    missing_columns = set(required_columns).difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}.")


def validate_temporal_order(values: pd.Series | pd.DataFrame) -> None:
    """Validate that a single-instrument input is chronologically ordered.

    A ``DatetimeIndex`` or ``PeriodIndex`` must be monotonically increasing. Other
    index types are left unchecked because they carry no temporal ordering
    guarantee.
    """
    index = values.index
    if isinstance(index, pd.DatetimeIndex | pd.PeriodIndex) and not index.is_monotonic_increasing:
        raise ValueError("values must be chronologically ordered before calculating this indicator")


def apply_by_ticker(
    values: pd.Series | pd.DataFrame,
    func: Callable[[pd.Series | pd.DataFrame], pd.Series | pd.DataFrame],
) -> pd.Series | pd.DataFrame:
    """Apply ``func`` per provider/ticker group for a multi-ticker input.

    ``values`` may be a ``pd.Series`` or a ``pd.DataFrame``. When it carries a
    ``MultiIndex`` the canonical market-price contract is enforced, the
    calculation is isolated within each provider/ticker group, and the original
    index is preserved. Otherwise ``values`` is treated as a single ordered
    sequence and ``func`` is applied after a temporal-order check.
    """
    if isinstance(values.index, pd.MultiIndex):
        validate_market_price_index(values)
        if values.empty:
            return func(values)
        results = [
            func(group) for _, group in values.groupby(level=["provider", "ticker"], sort=False)
        ]
        return pd.concat(results).reindex(values.index)

    validate_temporal_order(values)
    return func(values)
