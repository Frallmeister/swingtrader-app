import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features._numerical import (
    consecutive_true_count,
    linreg,
    safe_divide,
)


def test_safe_divide_masks_invalid_denominators_and_nonfinite_results() -> None:
    result = safe_divide(
        pd.Series([10.0, 10.0, 10.0, np.inf]),
        pd.Series([2.0, 0.0, np.inf, 2.0]),
    )

    pd.testing.assert_series_equal(result, pd.Series([5.0, np.nan, np.nan, np.nan]))


def test_safe_divide_requires_series_inputs() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        safe_divide(pd.Series([1.0]), 1.0)


def test_linreg_fits_a_perfect_line_to_its_endpoint() -> None:
    values = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])

    result = linreg(values, length=3, offset=0)

    # A perfectly linear window is recovered exactly, so the fitted value at the
    # newest position equals the observed value there. The first two rows warm up.
    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, np.nan, 5.0, 7.0, 9.0]),
    )


def test_linreg_offset_evaluates_an_earlier_position_in_the_window() -> None:
    values = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])

    result = linreg(values, length=3, offset=1)

    # offset=1 evaluates the middle of each three-row window, which for a perfect
    # line equals the middle observation.
    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, np.nan, 3.0, 5.0, 7.0]),
    )


def test_linreg_returns_the_constant_for_a_flat_series() -> None:
    values = pd.Series([4.0, 4.0, 4.0, 4.0])

    result = linreg(values, length=2, offset=0)

    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, 4.0, 4.0, 4.0]),
    )


def test_linreg_length_one_falls_back_to_the_observation() -> None:
    values = pd.Series([2.0, 5.0, 3.0])

    result = linreg(values, length=1, offset=0)

    pd.testing.assert_series_equal(result, values)


def test_linreg_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length must be at least 1"):
        linreg(pd.Series([1.0, 2.0]), length=0)


def test_linreg_matches_a_polyfit_reference() -> None:
    values = pd.Series([2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 9.0, 6.0])
    length = 4
    offset = 0

    result = linreg(values, length=length, offset=offset)

    x = np.arange(length, dtype=float)
    evaluation_position = length - 1 - offset
    expected = [np.nan] * len(values)
    for end in range(length - 1, len(values)):
        window = values.iloc[end - length + 1 : end + 1].to_numpy()
        slope, intercept = np.polyfit(x, window, 1)
        expected[end] = intercept + slope * evaluation_position

    pd.testing.assert_series_equal(result, pd.Series(expected), check_exact=False)


def test_consecutive_true_count_matches_the_documented_example() -> None:
    condition = pd.Series([pd.NA, False, True, True, True, False, True], dtype="boolean")

    result = consecutive_true_count(condition)

    pd.testing.assert_series_equal(
        result,
        pd.Series([pd.NA, 0, 1, 2, 3, 0, 1], dtype="Int64"),
    )


def test_consecutive_true_count_increments_over_an_unbroken_run() -> None:
    condition = pd.Series([True, True, True], dtype="boolean")

    result = consecutive_true_count(condition)

    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], dtype="Int64"))


def test_consecutive_true_count_is_zero_when_never_true() -> None:
    condition = pd.Series([False, False, False], dtype="boolean")

    result = consecutive_true_count(condition)

    pd.testing.assert_series_equal(result, pd.Series([0, 0, 0], dtype="Int64"))


def test_consecutive_true_count_breaks_the_run_on_missing_values() -> None:
    condition = pd.Series([True, pd.NA, True], dtype="boolean")

    result = consecutive_true_count(condition)

    # The missing observation breaks the run and stays missing; the following
    # true observation starts a fresh count.
    pd.testing.assert_series_equal(result, pd.Series([1, pd.NA, 1], dtype="Int64"))
