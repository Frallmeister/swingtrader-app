"""Broadly reusable, domain-neutral numerical helpers."""

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series and replace nonfinite or zero-denominator results with NA."""
    if not isinstance(numerator, pd.Series) or not isinstance(denominator, pd.Series):
        raise TypeError("Both numerator and denominator must be pandas Series objects.")

    valid_denominator = denominator.ne(0) & np.isfinite(denominator)
    result = numerator.div(denominator.where(valid_denominator))
    return result.where(np.isfinite(result))


def consecutive_true_count(condition: pd.Series) -> pd.Series:
    """Count consecutive true values, preserving missing observations."""
    condition = condition.astype("boolean")
    active = condition.fillna(False)
    run_id = (~active).cumsum()
    counts = active.astype("int64").groupby(run_id).cumsum().astype("Int64")
    return counts.where(condition.notna())
