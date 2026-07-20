import numpy as np
import pandas as pd

from swingtrader.indicators._smoothing import (
    exponential_moving_average,
    wilder_moving_average,
)


def test_exponential_moving_average_warms_up_before_window_fills() -> None:
    values = pd.Series([10.0, 11.0, 12.0, 13.0], name="adjusted_close")

    result = exponential_moving_average(values, length=3)

    # The first ``length - 1`` observations remain missing until the window fills.
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()


def test_exponential_moving_average_converges_to_a_constant_series() -> None:
    values = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0], name="adjusted_close")

    result = exponential_moving_average(values, length=3)

    # A constant series feeds an EMA of exactly that constant after warm-up.
    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, np.nan, 5.0, 5.0, 5.0], name="adjusted_close"),
    )


def test_exponential_moving_average_preserves_index() -> None:
    values = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.Index([2, 0, 1]),
        name="adjusted_close",
    )

    result = exponential_moving_average(values, length=2)

    pd.testing.assert_index_equal(result.index, values.index)


def test_wilder_moving_average_warms_up_before_window_fills() -> None:
    values = pd.Series([10.0, 11.0, 12.0, 13.0], name="adjusted_close")

    result = wilder_moving_average(values, length=3)

    # The first ``length - 1`` observations remain missing until the window fills.
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].notna().all()


def test_wilder_moving_average_converges_to_a_constant_series() -> None:
    values = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0], name="adjusted_close")

    result = wilder_moving_average(values, length=3)

    # A constant series feeds a Wilder average of exactly that constant after
    # warm-up.
    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, np.nan, 5.0, 5.0, 5.0], name="adjusted_close"),
    )


def test_wilder_moving_average_preserves_index() -> None:
    values = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.Index([2, 0, 1]),
        name="adjusted_close",
    )

    result = wilder_moving_average(values, length=2)

    pd.testing.assert_index_equal(result.index, values.index)
