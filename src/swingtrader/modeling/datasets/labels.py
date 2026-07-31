"""Target builders and execution helpers for canonical modeling datasets."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import (
    MARKET_PRICE_INDEX_NAMES,
    validate_market_price_index,
    validate_required_columns,
)

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
V1_RETURN_THRESHOLD = (1 + V1_COMMISSION + V1_REQUIRED_NET_RETURN) / (1 - V1_COMMISSION) - 1

# These are logical inputs to the versioned family manifest. The three market
# identifiers are supplied by the canonical index rather than ordinary columns.
REQUIRED_PRICE_COLUMNS = (*MARKET_PRICE_INDEX_NAMES, "adjusted_close")
FORWARD_RETURN_COLUMNS = tuple(
    f"forward_return_{horizon}d" for horizon in V1_FORWARD_RETURN_HORIZONS
)
TARGET_SIGNIFICANT_UP_5D_COLUMN = "target_significant_up_5d"


def add_forward_return_targets(
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Append adjusted-close returns over future observed sessions.

    ``prices`` must use the canonical, unique, sorted market-price MultiIndex
    with levels ``provider``, ``ticker``, and ``trading_date``. Returns are
    calculated independently within each provider/ticker series and the input
    index is preserved. A return is missing when either the current or required
    future adjusted close is unavailable, non-finite, or non-positive.
    """
    validate_market_price_index(prices)
    validate_required_columns(prices, required_columns={"adjusted_close"})

    result = prices.copy()
    adjusted_close = pd.to_numeric(result["adjusted_close"], errors="coerce").astype("float64")
    adjusted_close = adjusted_close.mask(adjusted_close.le(0) | ~np.isfinite(adjusted_close))
    grouped = adjusted_close.groupby(level=["provider", "ticker"], sort=False)
    for horizon in horizons:
        result[f"forward_return_{horizon}d"] = (
            grouped.shift(-horizon) / adjusted_close - 1
        ).astype("float64")
    return result


def add_fixed_return_target(
    data: pd.DataFrame,
    *,
    forward_return_column: str,
    output_column: str,
    threshold: float,
) -> pd.DataFrame:
    """Append a nullable Boolean target using a strict return threshold.

    The canonical market-price index is preserved. Missing forward returns stay
    missing rather than being coerced into the negative class.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={forward_return_column})

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
    """Return prices with the targets enforced by ``target_set`` appended."""
    return target_set.apply(prices)


def generate_v1_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate V1 labels for a canonical market-price DataFrame."""
    from swingtrader.modeling.datasets.catalog import V1_TARGET_SET

    return generate_target_set(prices, target_set=V1_TARGET_SET)


def generate_v3_labels(prices: pd.DataFrame) -> pd.DataFrame:
    """Generate V3 labels for a canonical market-price DataFrame."""
    from swingtrader.modeling.datasets.catalog import V3_TARGET_SET

    return generate_target_set(prices, target_set=V3_TARGET_SET)
