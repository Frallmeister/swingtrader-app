import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features._numerical import safe_divide


def test_safe_divide_masks_invalid_denominators_and_nonfinite_results() -> None:
    result = safe_divide(
        pd.Series([10.0, 10.0, 10.0, np.inf]),
        pd.Series([2.0, 0.0, np.inf, 2.0]),
    )

    pd.testing.assert_series_equal(result, pd.Series([5.0, np.nan, np.nan, np.nan]))


def test_safe_divide_requires_series_inputs() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        safe_divide(pd.Series([1.0]), 1.0)
