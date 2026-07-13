"""V1 forward-return labels for modeling datasets."""

from collections.abc import Sequence

import pandas as pd

V1_FORWARD_RETURN_HORIZONS = (5, 10, 15)
V1_COMMISSION = 0.0025
V1_ANNUAL_RETURN_TARGET = 0.50
V1_TRADING_DAYS_PER_YEAR = 252
V1_PREDICTION_HORIZON = 5

V1_REQUIRED_NET_RETURN = (1 + V1_ANNUAL_RETURN_TARGET) ** (
    V1_PREDICTION_HORIZON / V1_TRADING_DAYS_PER_YEAR
) - 1
V1_RETURN_THRESHOLD = (1 + V1_COMMISSION + V1_REQUIRED_NET_RETURN) / (1 - V1_COMMISSION) - 1

REQUIRED_PRICE_COLUMNS = ("provider", "ticker", "trading_date", "adjusted_close")
FORWARD_RETURN_COLUMNS = tuple(
    f"forward_return_{horizon}d" for horizon in V1_FORWARD_RETURN_HORIZONS
)
TARGET_SIGNIFICANT_UP_5D_COLUMN = "target_significant_up_5d"
LABEL_COLUMNS = (*FORWARD_RETURN_COLUMNS, TARGET_SIGNIFICANT_UP_5D_COLUMN)


def generate_v1_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Add V1 forward returns and the primary binary target to daily prices.

    The input must contain one row per ``provider``, ``ticker``, and ``trading_date``
    observation with an ``adjusted_close`` value. Horizons are measured in observed rows
    within each provider/ticker group, not calendar days. The returned DataFrame preserves
    the caller's rows and columns while appending label columns.

    Parameters
    ----------
    prices
        Daily price observations compatible with ``load_bronze_daily_prices()``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``prices`` with V1 label columns appended.

    Raises
    ------
    ValueError
        Raised when required columns are missing or duplicate provider/ticker/trading-date
        observations make forward-label alignment ambiguous.
    """
    _validate_required_columns(prices)

    labeled_prices = prices.copy()
    if labeled_prices.empty:
        return _append_empty_label_columns(labeled_prices)

    normalized_dates = pd.to_datetime(labeled_prices["trading_date"])
    _validate_unique_observations(labeled_prices, normalized_dates)

    calculation_frame = pd.DataFrame(
        {
            "__original_index": range(len(labeled_prices)),
            "provider": labeled_prices["provider"].to_numpy(),
            "ticker": labeled_prices["ticker"].to_numpy(),
            "trading_date": normalized_dates.to_numpy(),
            "adjusted_close": pd.to_numeric(labeled_prices["adjusted_close"], errors="coerce"),
        }
    )
    calculation_frame = calculation_frame.sort_values(
        ["provider", "ticker", "trading_date", "__original_index"],
        kind="mergesort",
    )

    grouped_adjusted_close = calculation_frame.groupby(
        ["provider", "ticker"],
        sort=False,
    )["adjusted_close"]

    for horizon in V1_FORWARD_RETURN_HORIZONS:
        forward_adjusted_close = grouped_adjusted_close.shift(-horizon)
        calculation_frame[f"forward_return_{horizon}d"] = (
            forward_adjusted_close / calculation_frame["adjusted_close"] - 1
        )

    calculation_frame = calculation_frame.sort_values("__original_index", kind="mergesort")
    calculation_frame.index = labeled_prices.index

    for column in FORWARD_RETURN_COLUMNS:
        labeled_prices[column] = calculation_frame[column].astype("float64")

    target = pd.Series(pd.NA, index=labeled_prices.index, dtype="boolean")
    valid_forward_return_5d = labeled_prices["forward_return_5d"].notna()
    target.loc[valid_forward_return_5d] = (
        labeled_prices.loc[valid_forward_return_5d, "forward_return_5d"] > V1_RETURN_THRESHOLD
    ).astype("boolean")
    labeled_prices[TARGET_SIGNIFICANT_UP_5D_COLUMN] = target

    return labeled_prices


def _validate_required_columns(prices: pd.DataFrame) -> None:
    missing_columns = _missing_columns(REQUIRED_PRICE_COLUMNS, prices.columns)
    if not missing_columns:
        return

    msg = f"Missing required price columns: {', '.join(missing_columns)}"
    raise ValueError(msg)


def _validate_unique_observations(
    prices: pd.DataFrame,
    normalized_dates: pd.Series,
) -> None:
    keys = pd.DataFrame(
        {
            "provider": prices["provider"],
            "ticker": prices["ticker"],
            "trading_date": normalized_dates,
        }
    )
    if not keys.duplicated().any():
        return

    msg = "Duplicate provider/ticker/trading_date observations are not allowed"
    raise ValueError(msg)


def _append_empty_label_columns(prices: pd.DataFrame) -> pd.DataFrame:
    labeled_prices = prices.copy()
    for column in FORWARD_RETURN_COLUMNS:
        labeled_prices[column] = pd.Series(dtype="float64")
    labeled_prices[TARGET_SIGNIFICANT_UP_5D_COLUMN] = pd.Series(dtype="boolean")
    return labeled_prices


def _missing_columns(required_columns: Sequence[str], available_columns: pd.Index) -> list[str]:
    return [column for column in required_columns if column not in available_columns]
