"""Shared price-adjustment helpers for model-facing feature builders."""

from collections.abc import Sequence

import pandas as pd

from swingtrader.core.numerical import safe_divide


def adjustment_consistent_price_frame(
    data: pd.DataFrame,
    *,
    price_columns: Sequence[str],
) -> pd.DataFrame:
    """Return selected price columns expressed on the adjusted-close scale.

    Each requested raw price column is multiplied by the row-wise factor
    ``adjusted_close / close``. When ``close`` is requested, valid rows are
    assigned directly from ``adjusted_close`` so the result exactly matches the
    feature layer's reference series. Rows with an invalid adjustment factor
    remain missing. The input dataframe is not mutated.

    Public feature builders validate their own required columns before calling
    this private helper. Reusable indicators remain source-agnostic and operate
    on whichever price representation their caller supplies.
    """
    columns = list(price_columns)
    adjustment_factor = safe_divide(
        data.loc[:, "adjusted_close"],
        data.loc[:, "close"],
    )
    result = data.loc[:, columns].mul(adjustment_factor, axis=0)
    if "close" in result.columns:
        result.loc[:, "close"] = data.loc[:, "adjusted_close"].where(
            adjustment_factor.notna()
        )
    return result
