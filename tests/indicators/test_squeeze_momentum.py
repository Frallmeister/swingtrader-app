import numpy as np
import pandas as pd
import pytest

from swingtrader.core.numerical import safe_divide
from swingtrader.indicators.moving_averages import sma
from swingtrader.indicators.squeeze_momentum import (
    _consecutive_true_count,
    _linreg,
    lazybear_squeeze_momentum,
)
from swingtrader.indicators.volatility import _atr, _true_range


def test_lazybear_squeeze_momentum_returns_expected_columns_and_preserves_index() -> None:
    ohlc = _squeeze_ohlc()

    result = lazybear_squeeze_momentum(ohlc, bb_length=5, kc_length=5, atr_length=5)

    assert list(result.columns) == [
        "squeeze_on",
        "squeeze_off",
        "squeeze_released",
        "squeeze_width_ratio",
        "squeeze_momentum",
        "squeeze_momentum_atr",
        "squeeze_momentum_atr_change",
        "squeeze_duration",
        "squeeze_release_duration",
    ]
    pd.testing.assert_index_equal(result.index, ohlc.index)


def test_lazybear_squeeze_momentum_matches_reference_definition() -> None:
    ohlc = _squeeze_ohlc()

    result = lazybear_squeeze_momentum(
        ohlc, bb_length=5, bb_mult=2.0, kc_length=5, kc_mult=1.5, atr_length=5
    )
    expected = _reference_squeeze(
        ohlc, bb_length=5, bb_mult=2.0, kc_length=5, kc_mult=1.5, atr_length=5
    )

    pd.testing.assert_frame_equal(result, expected)


def test_lazybear_squeeze_momentum_computes_true_range_and_atr_internally() -> None:
    ohlc = _squeeze_ohlc()
    # Precomputed true_range/atr columns are neither required nor used; a frame
    # carrying deliberately wrong values yields the same result as one without.
    polluted = ohlc.copy()
    polluted["true_range"] = -999.0
    polluted["atr"] = -999.0

    result = lazybear_squeeze_momentum(ohlc, bb_length=5, kc_length=5, atr_length=5)
    from_polluted = lazybear_squeeze_momentum(polluted, bb_length=5, kc_length=5, atr_length=5)

    pd.testing.assert_frame_equal(result, from_polluted)


def test_lazybear_squeeze_momentum_uses_bb_mult_not_kc_mult() -> None:
    ohlc = _squeeze_ohlc()
    close = ohlc["close"]
    true_range_ = _true_range(ohlc)

    result = lazybear_squeeze_momentum(
        ohlc, bb_length=5, bb_mult=2.0, kc_length=5, kc_mult=1.5, atr_length=5
    )

    # The band width ratio is (bb_mult * std) / (kc_mult * range_ma). Had the
    # Bollinger deviation reused kc_mult, the multipliers would cancel and the
    # ratio would collapse to std / range_ma.
    std = close.rolling(window=5, min_periods=5).std(ddof=0)
    range_ma = sma(true_range_, length=5)
    expected_ratio = safe_divide(2.0 * std, 1.5 * range_ma).rename("squeeze_width_ratio")
    collapsed_ratio = safe_divide(std, range_ma)

    pd.testing.assert_series_equal(result["squeeze_width_ratio"], expected_ratio)
    assert not np.allclose(
        result["squeeze_width_ratio"].dropna(),
        collapsed_ratio.reindex(result.index).dropna(),
    )


def test_lazybear_squeeze_momentum_state_matches_band_nesting() -> None:
    ohlc = _squeeze_ohlc()

    result = lazybear_squeeze_momentum(ohlc, bb_length=5, kc_length=5, atr_length=5)

    # The engineered mid-series low-volatility segment must trigger at least one
    # squeeze, and on/off are mutually exclusive wherever both are defined.
    assert (result["squeeze_on"] == True).any()  # noqa: E712
    both_defined = result["squeeze_on"].notna() & result["squeeze_off"].notna()
    overlap = result.loc[both_defined, "squeeze_on"] & result.loc[both_defined, "squeeze_off"]
    assert not overlap.any()


def test_lazybear_squeeze_momentum_derived_columns_are_consistent() -> None:
    ohlc = _squeeze_ohlc()

    result = lazybear_squeeze_momentum(ohlc, bb_length=5, kc_length=5, atr_length=5)

    squeeze_on = result["squeeze_on"]
    expected_duration = _consecutive_true_count(squeeze_on).rename("squeeze_duration")
    expected_released = (squeeze_on.shift(1, fill_value=False) & squeeze_on.eq(False)).rename(
        "squeeze_released"
    )
    expected_release_duration = (
        expected_duration.shift(1).where(expected_released).rename("squeeze_release_duration")
    )
    expected_momentum_atr_change = (
        result["squeeze_momentum_atr"].diff().rename("squeeze_momentum_atr_change")
    )

    pd.testing.assert_series_equal(result["squeeze_duration"], expected_duration)
    pd.testing.assert_series_equal(result["squeeze_released"], expected_released)
    pd.testing.assert_series_equal(result["squeeze_release_duration"], expected_release_duration)
    pd.testing.assert_series_equal(
        result["squeeze_momentum_atr_change"], expected_momentum_atr_change
    )


def test_lazybear_squeeze_momentum_warms_up_before_windows_fill() -> None:
    ohlc = _squeeze_ohlc()

    result = lazybear_squeeze_momentum(ohlc, bb_length=5, kc_length=5, atr_length=5)

    # The squeeze state is defined once the five-row band windows fill (row 4).
    assert result["squeeze_on"].iloc[:4].isna().all()
    assert result["squeeze_on"].iloc[4:].notna().all()
    # The momentum histogram warms up later: the detrended close is itself only
    # defined from row 4, and the linear regression then needs a further five-row
    # window, so the first defined value lands at row 2 * kc_length - 2 (row 8).
    assert result["squeeze_momentum"].iloc[:8].isna().all()
    assert result["squeeze_momentum"].iloc[8:].notna().all()


def test_lazybear_squeeze_momentum_isolates_tickers() -> None:
    multi = _squeeze_multi()

    result = lazybear_squeeze_momentum(multi, bb_length=5, kc_length=5, atr_length=5)

    pd.testing.assert_index_equal(result.index, multi.index)
    aaa = lazybear_squeeze_momentum(
        multi.loc[("yfinance", "AAA.ST")], bb_length=5, kc_length=5, atr_length=5
    )
    pd.testing.assert_frame_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        aaa.reset_index(drop=True),
    )


def test_lazybear_squeeze_momentum_requires_price_columns() -> None:
    ohlc = _squeeze_ohlc().drop(columns="high")

    with pytest.raises(ValueError, match="Missing required columns"):
        lazybear_squeeze_momentum(ohlc)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bb_length": 0}, "positive integer"),
        ({"kc_length": -1}, "positive integer"),
        ({"atr_length": 0}, "positive integer"),
        ({"bb_mult": 0.0}, "positive number"),
        ({"kc_mult": -1.5}, "positive number"),
    ],
)
def test_lazybear_squeeze_momentum_rejects_invalid_configuration(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        lazybear_squeeze_momentum(_squeeze_ohlc(), **kwargs)


def test_linreg_fits_a_perfect_line_to_its_endpoint() -> None:
    values = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])

    result = _linreg(values, length=3, offset=0)

    # A perfectly linear window is recovered exactly, so the fitted value at the
    # newest position equals the observed value there. The first two rows warm up.
    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, np.nan, 5.0, 7.0, 9.0]),
    )


def test_linreg_offset_evaluates_an_earlier_position_in_the_window() -> None:
    values = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0])

    result = _linreg(values, length=3, offset=1)

    # offset=1 evaluates the middle of each three-row window, which for a perfect
    # line equals the middle observation.
    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, np.nan, 3.0, 5.0, 7.0]),
    )


def test_linreg_returns_the_constant_for_a_flat_series() -> None:
    values = pd.Series([4.0, 4.0, 4.0, 4.0])

    result = _linreg(values, length=2, offset=0)

    pd.testing.assert_series_equal(
        result,
        pd.Series([np.nan, 4.0, 4.0, 4.0]),
    )


def test_linreg_length_one_falls_back_to_the_observation() -> None:
    values = pd.Series([2.0, 5.0, 3.0])

    result = _linreg(values, length=1, offset=0)

    pd.testing.assert_series_equal(result, values)


def test_linreg_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="length must be at least 1"):
        _linreg(pd.Series([1.0, 2.0]), length=0)


def test_linreg_matches_a_polyfit_reference() -> None:
    values = pd.Series([2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 9.0, 6.0])
    length = 4
    offset = 0

    result = _linreg(values, length=length, offset=offset)

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

    result = _consecutive_true_count(condition)

    pd.testing.assert_series_equal(
        result,
        pd.Series([pd.NA, 0, 1, 2, 3, 0, 1], dtype="Int64"),
    )


def test_consecutive_true_count_increments_over_an_unbroken_run() -> None:
    condition = pd.Series([True, True, True], dtype="boolean")

    result = _consecutive_true_count(condition)

    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3], dtype="Int64"))


def test_consecutive_true_count_is_zero_when_never_true() -> None:
    condition = pd.Series([False, False, False], dtype="boolean")

    result = _consecutive_true_count(condition)

    pd.testing.assert_series_equal(result, pd.Series([0, 0, 0], dtype="Int64"))


def test_consecutive_true_count_breaks_the_run_on_missing_values() -> None:
    condition = pd.Series([True, pd.NA, True], dtype="boolean")

    result = _consecutive_true_count(condition)

    # The missing observation breaks the run and stays missing; the following
    # true observation starts a fresh count.
    pd.testing.assert_series_equal(result, pd.Series([1, pd.NA, 1], dtype="Int64"))


def _reference_squeeze(
    ohlc: pd.DataFrame,
    *,
    bb_length: int,
    bb_mult: float,
    kc_length: int,
    kc_mult: float,
    atr_length: int,
) -> pd.DataFrame:
    close = ohlc["close"]
    high = ohlc["high"]
    low = ohlc["low"]
    true_range_ = _true_range(ohlc)
    atr_ = _atr(ohlc, length=atr_length)

    bb_basis = sma(close, length=bb_length)
    bb_deviation = bb_mult * close.rolling(window=bb_length, min_periods=bb_length).std(ddof=0)
    upper_bb = bb_basis + bb_deviation
    lower_bb = bb_basis - bb_deviation

    kc_basis = sma(close, length=kc_length)
    range_ma = sma(true_range_, length=kc_length)
    upper_kc = kc_basis + kc_mult * range_ma
    lower_kc = kc_basis - kc_mult * range_ma

    squeeze_ready = pd.concat([upper_bb, lower_bb, upper_kc, lower_kc], axis=1).notna().all(axis=1)
    squeeze_on = (
        ((lower_bb > lower_kc) & (upper_bb < upper_kc)).astype("boolean").where(squeeze_ready)
    )
    squeeze_off = (
        ((lower_bb < lower_kc) & (upper_bb > upper_kc)).astype("boolean").where(squeeze_ready)
    )
    squeeze_width_ratio = safe_divide(upper_bb - lower_bb, upper_kc - lower_kc)
    squeeze_released = squeeze_on.shift(1, fill_value=False) & squeeze_on.eq(False)
    squeeze_duration = _consecutive_true_count(squeeze_on)
    squeeze_release_duration = squeeze_duration.shift(1).where(squeeze_released)

    highest_high = high.rolling(window=kc_length, min_periods=kc_length).max()
    lowest_low = low.rolling(window=kc_length, min_periods=kc_length).min()
    range_midpoint = (highest_high + lowest_low) / 2.0
    reference_level = (range_midpoint + kc_basis) / 2.0
    momentum = _linreg(close - reference_level, length=kc_length, offset=0)
    momentum_atr = safe_divide(momentum, atr_)

    return pd.DataFrame(
        {
            "squeeze_on": squeeze_on,
            "squeeze_off": squeeze_off,
            "squeeze_released": squeeze_released,
            "squeeze_width_ratio": squeeze_width_ratio,
            "squeeze_momentum": momentum,
            "squeeze_momentum_atr": momentum_atr,
            "squeeze_momentum_atr_change": momentum_atr.diff(),
            "squeeze_duration": squeeze_duration,
            "squeeze_release_duration": squeeze_release_duration,
        },
        index=ohlc.index,
    )


def _squeeze_ohlc(*, include_adjusted_and_volume: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 40
    steps = rng.normal(0.0, 1.0, n)
    # A calm middle segment compresses volatility to engineer a squeeze between
    # two more volatile stretches.
    scale = np.concatenate([np.full(12, 1.5), np.full(16, 0.2), np.full(n - 28, 1.5)])
    close = 100.0 + np.cumsum(steps * scale)
    span = np.abs(rng.normal(0.0, 1.0, n)) * scale + 0.1
    index = pd.DatetimeIndex(pd.date_range("2026-01-01", periods=n, freq="B"), name="trading_date")

    columns = {"high": close + span, "low": close - span, "close": close}
    if include_adjusted_and_volume:
        columns["adjusted_close"] = close
        columns["volume"] = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame(columns, index=index)


def _squeeze_multi(*, include_adjusted_and_volume: bool = False) -> pd.DataFrame:
    single = _squeeze_ohlc(include_adjusted_and_volume=include_adjusted_and_volume)
    trading_dates = [timestamp.date() for timestamp in single.index]

    frames = []
    for ticker, price_shift in (("AAA.ST", 0.0), ("BBB.ST", 40.0)):
        frame = single.copy()
        for column in ("high", "low", "close"):
            frame[column] = frame[column] + price_shift
        if include_adjusted_and_volume:
            frame["adjusted_close"] = frame["close"]
        frame.index = pd.MultiIndex.from_arrays(
            [["yfinance"] * len(frame), [ticker] * len(frame), trading_dates],
            names=["provider", "ticker", "trading_date"],
        )
        frames.append(frame)
    return pd.concat(frames).sort_index()
