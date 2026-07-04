"""Shared validation helpers for ingestion workflows.

The helpers keep common argument validation consistent across ingestion libraries and runnable
jobs without coupling those modules to each other's implementation details.
"""

from datetime import date


def validate_date_window(*, start_date: date, end_date: date) -> None:
    """Validate an inclusive start date and exclusive end date.

    Parameters
    ----------
    start_date
        Inclusive start date.
    end_date
        Exclusive end date.

    Raises
    ------
    ValueError
        Raised when ``start_date`` is not before ``end_date``.
    """
    if start_date >= end_date:
        raise ValueError("start_date must be before end_date.")


def validate_limit(limit: int | None) -> None:
    """Validate an optional positive ticker limit.

    Parameters
    ----------
    limit
        Optional maximum number of tickers to process.

    Raises
    ------
    ValueError
        Raised when ``limit`` is less than one.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero.")
