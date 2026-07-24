"""Forward-return and fixed-threshold target builders."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from swingtrader.modeling.datasets.contracts import TargetSetSpec

V1_FORWARD_RETURN_HORIZONS = (5, 10, 15)
V1_COMMISSION = 0.0025
V1_ANNUAL_RETURN_TARGET = 0.50
V1_TRADING_DAYS_PER_YEAR = 252
V1_PREDICTION_HORIZON = 5
V1_REQUIRED_NET_RETURN = (1 + V1_ANNUAL_RETURN_TARGET) ** (
    V1_PREDICTION_HORIZON / V1_TRADING_DAYS_PER_YEAR
) - 1
V1_RETURN_THRESHOLD = (1 + V1_COMMISSION + V1_REQUIRED_NET_RETURN) / (
    1 - V1_COMMISSION
) - 1

REQUIRED_PRICE_COLUMNS = ("provider", "ticker", "trading_date", "adjusted_close")
FORWARD_RETURN_COLUMNS = tuple(
    f"forward_return_{horizon}d" for horizon in V1_FORWARD_RETURN_HORIZONS
)
TARGET_SIGNIFICANT_UP_5D_COLUMN = "target_significant_up_5d"
LABEL_COLUMNS = (*FORWARD_RETURN_COLUMNS, TARGET_SIGNIFICANT_UP_5D_COLUMN)


def add_forward_return_targets(
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Append adjusted-close forward returns for observed-session horizons."""
    _validate_required_columns(prices)
    result = prices.copy()
    if result.empty:
        for horizon in horizons:
            result[f"forward_return_{horizon}d"] = pd.Series(dtype="float64")
        return result

    normalized_dates = pd.to_datetime(result["trading_date"])
    _validate_unique_observations(result, normalized_dates)
    calculation_frame = pd.DataFrame(
        {
            "__original_index": range(len(result)),
            "provider": result["provider"].to_numpy(),
            "ticker": result["ticker"].to_numpy(),
            "trading_date": normalized_dates.to_numpy(),
            "adjusted_close": pd.to_numeric(result["adjusted_close"], errors="coerce"),
        }
    )
    adjusted_close = calculation_frame["adjusted_close"]
    calculation_frame["adjusted_close"] = adjusted_close.mask(
        adjusted_close.le(0) | ~np.isfinite(adjusted_close)
    )
    calculation_frame = calculation_frame.sort_values(
        ["provider", "ticker", "trading_date", "__original_index"], kind="mergesort"
    )
    grouped = calculation_frame.groupby(["provider", "ticker"], sort=False)["adjusted_close"]
    for horizon in horizons:
        calculation_frame[f"forward_return_{horizon}d"] = (
            grouped.shift(-horizon) / calculation_frame["adjusted_close"] - 1
        )
    calculation_frame = calculation_frame.sort_values("__original_index", kind="mergesort")
    calculation_frame.index = result.index
    for horizon in horizons:
        column = f"forward_return_{horizon}d"
        result[column] = calculation_frame[column].astype("float64")
    return result


def add_fixed_return_target(
    data: pd.DataFrame,
    *,
    forward_return_column: str,
    output_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Append a nullable Boolean target using a strict return threshold."""
    if forward_return_column not in data.columns:
        raise ValueError(f"Missing required target column: {forward_return_column}")
    result = data.copy()
    target = pd.Series(pd.NA, index=result.index, dtype="boolean")
    valid = result[forward_return_column].notna()
    target.loc[valid] = result.loc[valid, forward_return_column].gt(threshold).astype("boolean")
    result[output_column] = target
    return result


def generate_target_set(
    prices: pd.DataFrame,
    *,
    target_set: "TargetSetSpec",
) -> pd.DataFrame:
    """Execute target families in declared order with schema validation."""
    result = prices
    for family in target_set.families:
        missing = sorted(family.required_columns.difference(result.columns))
        if missing:
            raise ValueError(
                f"Target family {family.name!r} is missing required columns: {', '.join(missing)}"
            )
        collisions = sorted(set(family.output_columns).intersection(result.columns))
        if collisions:
            raise ValueError(
                f"Target family {family.name!r} would overwrite columns: {', '.join(collisions)}"
            )
        result = family.apply(result)
        missing_outputs = sorted(set(family.output_columns).difference(result.columns))
        if missing_outputs:
            raise ValueError(
                f"Target family {family.name!r} did not produce columns: "
                f"{', '.join(missing_outputs)}"
            )
    return result


def generate_v1_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Add the versioned V1 target set while preserving historical behavior."""
    from swingtrader.modeling.datasets.target_catalog import V1_TARGET_SET

    return generate_target_set(prices, target_set=V1_TARGET_SET)


def _validate_required_columns(prices: pd.DataFrame) -> None:
    missing_columns = _missing_columns(REQUIRED_PRICE_COLUMNS, prices.columns)
    if missing_columns:
        raise ValueError(f"Missing required price columns: {', '.join(missing_columns)}")


def _validate_unique_observations(prices: pd.DataFrame, normalized_dates: pd.Series) -> None:
    keys = pd.DataFrame(
        {
            "provider": prices["provider"],
            "ticker": prices["ticker"],
            "trading_date": normalized_dates,
        }
    )
    if keys.duplicated().any():
        raise ValueError("Duplicate provider/ticker/trading_date observations are not allowed")


def _missing_columns(required_columns: Sequence[str], available_columns: pd.Index) -> list[str]:
    return [column for column in required_columns if column not in available_columns]
