"""
Helper functions for numerical calculations in features.
"""

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    if not isinstance(numerator, pd.Series) or not isinstance(denominator, pd.Series):
        raise TypeError("Both numerator and denominator must be pandas Series objects.")

    valid_denominator = denominator.ne(0) & np.isfinite(denominator)
    result = numerator.div(
        denominator.where(valid_denominator)
    )
    return result.where(np.isfinite(result))
