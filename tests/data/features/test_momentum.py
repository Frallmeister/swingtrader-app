import numpy as np
import pandas as pd
import pytest

import swingtrader.data.features.momentum as momentum_module
from swingtrader.data.features._numerical import (
    consecutive_true_count,
    linreg,
    safe_divide,
)
from swingtrader.data.features.momentum import (
    add_momentum_features,
    lazybear_squeeze_momentum,
    macd,
    mfi,
    ppo,
    ppo_percentile,
    rsi,
    stochastic_oscillator,
)
from swingtrader.data.features.trends import sma
from swingtrader.data.features.volatility import _atr, _true_range, bollinger_percent_b


def test_add_momentum_features_preserves_source_columns_and_adds_final_features() -> None:
    prices = _indexed_prices()
    original = prices.copy(deep=True)

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    expected_columns = [
        *prices.columns,
        "ppo",
        "ppo_signal",
        "ppo_histogram",
        "ppo_percentile",
        "rsi",
        "rsi_percent_b",
        "stochastic_k",
        "stochastic_d",
        "mfi",
        "mfi_percent_b",
        "squeeze_on",
        "squeeze_off",
        "squeeze_released",
        "squeeze_width_ratio",
        "squeeze_momentum_atr",
        "squeeze_momentum_atr_change",
        "squeeze_duration",
        "squeeze_release_duration",
    ]
    assert list(result.columns) == expected_columns
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    expected_ppo = ppo(prices["adjusted_close"], lengths=(2, 3, 2), use_percent=False)
    pd.testing.assert_series_equal(result["ppo"], expected_ppo["ppo"], check_exact=False)
    pd.testing.assert_series_equal(
        result["ppo_signal"], expected_ppo["ppo_signal"], check_exact=False
    )
    pd.testing.assert_series_equal(
        result["ppo_histogram"],
        result["ppo"].sub(result["ppo_signal"]).rename("ppo_histogram"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ppo_percentile"].reset_index(drop=True),
        pd.Series(
            [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
            name="ppo_percentile",
        ),
        check_exact=False,
    )


def test_add_momentum_features_uses_custom_ppo_lengths() -> None:
    prices = _indexed_prices()

    default_lengths = add_momentum_features(prices, ppo_percentile_min_history=1)
    custom_lengths = add_momentum_features(
        prices, ppo_lengths=(2, 3, 2), ppo_percentile_min_history=1
    )

    # The default 12/26/9 windows never warm up on this short history, while the
    # short custom windows produce populated PPO values.
    assert default_lengths["ppo"].notna().sum() == 0
    assert custom_lengths["ppo"].notna().sum() > 0


def test_add_momentum_features_preserves_multiindex_and_calculates_each_ticker() -> None:
    prices = _indexed_prices()

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    pd.testing.assert_index_equal(result.index, prices.index)
    # Each ticker warms up independently; a constant BBB.ST price yields a zero
    # PPO after warm-up, isolated from AAA.ST.
    bbb_ppo = result.loc[("yfinance", "BBB.ST"), "ppo"]
    assert (bbb_ppo.dropna() == 0.0).all()
    assert result.loc[("yfinance", "AAA.ST"), "ppo"].notna().sum() == 2
    assert result.loc[("yfinance", "BBB.ST"), "ppo"].notna().sum() == 2


def test_add_momentum_features_rejects_identifiers_as_columns() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="MultiIndex with levels"):
        add_momentum_features(prices, ppo_lengths=(2, 3, 2))


def test_add_momentum_features_rejects_unsorted_input() -> None:
    prices = _indexed_prices().iloc[[1, 0, 2, 3, 4, 5, 6, 7]]

    with pytest.raises(ValueError, match="must be sorted"):
        add_momentum_features(prices, ppo_lengths=(2, 3, 2))


def test_add_momentum_features_requires_adjusted_close() -> None:
    prices = _indexed_prices().drop(columns="adjusted_close")

    with pytest.raises(ValueError, match="Missing required columns"):
        add_momentum_features(prices)


def test_add_momentum_features_rejects_invalid_configuration() -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="fast length"):
        add_momentum_features(prices, ppo_lengths=(3, 2, 1))

    with pytest.raises(ValueError, match="positive integer"):
        add_momentum_features(prices, ppo_percentile_min_history=0)


def test_add_momentum_features_delegates_to_ppo(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _indexed_prices()
    calls = 0

    def fake_ppo(
        values: pd.Series, *, lengths: tuple[int, int, int] = (12, 26, 9), use_percent: bool = True
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        assert lengths == (2, 3, 2)
        assert use_percent is False
        return pd.DataFrame(
            {
                "ppo": [0.0] * len(values),
                "ppo_signal": [0.0] * len(values),
                "ppo_histogram": [0.0] * len(values),
            },
            index=values.index,
        )

    monkeypatch.setattr(momentum_module, "ppo", fake_ppo)

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    # ``ppo`` already isolates provider/ticker groups internally, so the
    # orchestrator delegates to it once rather than grouping the prices itself.
    assert calls == 1
    assert list(result.loc[:, ["ppo", "ppo_signal", "ppo_histogram"]].columns) == [
        "ppo",
        "ppo_signal",
        "ppo_histogram",
    ]


def test_ppo_returns_percent_by_default_or_ratio_when_requested() -> None:
    values = _prices()["adjusted_close"].iloc[:4]

    ratio = ppo(values, lengths=(2, 3, 2), use_percent=False)
    percent = ppo(values, lengths=(2, 3, 2))

    pd.testing.assert_frame_equal(percent, ratio.mul(100))


def test_ppo_returns_dataframe_with_expected_columns_and_values() -> None:
    prices = _prices()["adjusted_close"].iloc[:4]

    result = ppo(prices, lengths=(2, 3, 2), use_percent=False)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["ppo", "ppo_signal", "ppo_histogram"]
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_series_equal(
        result["ppo"],
        pd.Series(
            [np.nan, np.nan, 0.04888888888888887, 0.05523099415204676],
            name="ppo",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ppo_signal"],
        result["ppo"].ewm(span=2, adjust=False, min_periods=2).mean().rename("ppo_signal"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ppo_histogram"],
        result["ppo"].sub(result["ppo_signal"]).rename("ppo_histogram"),
        check_exact=False,
    )


def test_ppo_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    result = ppo(values, lengths=(1, 2, 1))

    pd.testing.assert_index_equal(result.index, values.index)


@pytest.mark.parametrize("lengths", [(0, 3, 2), (True, 3, 2), (2, 3, 0), (2, 3, 1.5)])
def test_ppo_rejects_invalid_lengths(lengths: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ppo(
            _prices()["adjusted_close"].iloc[:4],
            lengths=lengths,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("lengths", [(2, 2, 1), (3, 2, 1)])
def test_ppo_rejects_fast_length_greater_than_or_equal_to_slow(
    lengths: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="fast length"):
        ppo(_prices()["adjusted_close"].iloc[:4], lengths=lengths)


def test_ppo_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = ppo(close, lengths=(1, 2, 1))

    assert list(result.columns) == ["ppo", "ppo_signal", "ppo_histogram"]
    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price yields a PPO of zero after warm-up, isolated from AAA.ST.
    bbb_ppo = result.loc[("yfinance", "BBB.ST"), "ppo"]
    assert (bbb_ppo.dropna() == 0.0).all()
    assert bbb_ppo.isna().sum() == 1


def test_ppo_percentile_calculates_grouped_point_in_time_rank() -> None:
    data = pd.Series([1.0, 3.0, 2.0, 2.0], name="ppo")

    percentile = ppo_percentile(data, min_history=1)

    expected = pd.Series(
        [np.nan, 1.0, 0.5, 2.0 / 3.0],
        name="ppo",
    )
    pd.testing.assert_series_equal(percentile, expected, check_exact=False)


@pytest.mark.parametrize("min_history", [True, 0, -1, 1.5])
def test_ppo_percentile_rejects_invalid_min_history(min_history: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ppo_percentile(
            pd.Series([1.0, 2.0], name="ppo"),
            min_history=min_history,  # type: ignore[arg-type]
        )


def test_ppo_percentile_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([1.0, 3.0, 2.0], index=pd.Index([2, 0, 1]), name="ppo")

    result = ppo_percentile(values, min_history=1)

    pd.testing.assert_index_equal(result.index, values.index)


def test_ppo_percentile_groups_by_ticker_index_levels() -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * 6,
            ["AAA.ST", "AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST", "BBB.ST"],
            pd.to_datetime(
                ["2026-07-01", "2026-07-02", "2026-07-03"] * 2,
            ),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    values = pd.Series([1.0, 3.0, 2.0, 10.0, 5.0, 7.0], index=index, name="ppo")

    result = ppo_percentile(values, min_history=1)

    pd.testing.assert_index_equal(result.index, values.index)
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        pd.Series([np.nan, 1.0, 0.5], name="ppo"),
    )
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series([np.nan, 0.0, 0.5], name="ppo"),
    )


def test_macd_returns_dataframe_with_expected_columns_and_values() -> None:
    prices = _prices()["adjusted_close"].iloc[:4]

    result = macd(prices, lengths=(2, 3, 2))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["macd", "macd_signal", "macd_histogram"]
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_series_equal(
        result["macd"],
        pd.Series(
            [np.nan, np.nan, 0.6111111111111112, 0.7870370370370372],
            name="macd",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["macd_signal"],
        result["macd"].ewm(span=2, adjust=False, min_periods=2).mean().rename("macd_signal"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["macd_histogram"],
        result["macd"].sub(result["macd_signal"]).rename("macd_histogram"),
        check_exact=False,
    )


def test_macd_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    result = macd(values, lengths=(1, 2, 1))

    pd.testing.assert_index_equal(result.index, values.index)


@pytest.mark.parametrize("lengths", [(0, 3, 2), (True, 3, 2), (2, 3, 0), (2, 3, 1.5)])
def test_macd_rejects_invalid_lengths(lengths: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        macd(
            _prices()["adjusted_close"].iloc[:4],
            lengths=lengths,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("lengths", [(2, 2, 1), (3, 2, 1)])
def test_macd_rejects_fast_length_greater_than_or_equal_to_slow(
    lengths: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="fast length"):
        macd(_prices()["adjusted_close"].iloc[:4], lengths=lengths)


def test_macd_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = macd(close, lengths=(1, 2, 1))

    assert list(result.columns) == ["macd", "macd_signal", "macd_histogram"]
    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price yields a MACD of zero after warm-up, isolated from AAA.ST.
    bbb_macd = result.loc[("yfinance", "BBB.ST"), "macd"]
    assert (bbb_macd.dropna() == 0.0).all()
    assert bbb_macd.isna().sum() == 1


def test_add_momentum_features_adds_rsi_from_adjusted_close_and_rsi_percent_b() -> None:
    prices = _indexed_prices()

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
        rsi_length=2,
        rsi_bollinger_length=2,
    )

    expected_rsi = rsi(prices["adjusted_close"], length=2)
    pd.testing.assert_series_equal(result["rsi"], expected_rsi.rename("rsi"), check_exact=False)

    expected_percent_b = bollinger_percent_b(result["rsi"], length=2, num_std=2.0)
    pd.testing.assert_series_equal(
        result["rsi_percent_b"],
        expected_percent_b.rename("rsi_percent_b"),
        check_exact=False,
    )
    # AAA.ST rises every day, so once the window warms up its RSI is a pure 100.
    aaa_rsi = result.loc[("yfinance", "AAA.ST"), "rsi"]
    assert (aaa_rsi.dropna() == 100.0).all()
    # BBB.ST is flat, so it has neither gains nor losses and RSI stays missing.
    assert result.loc[("yfinance", "BBB.ST"), "rsi"].isna().all()


def test_add_momentum_features_rejects_invalid_rsi_length() -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="positive integer"):
        add_momentum_features(prices, ppo_lengths=(2, 3, 2), rsi_length=0)


def test_rsi_is_100_without_losses_and_0_without_gains() -> None:
    rising = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="adjusted_close")
    falling = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], name="adjusted_close")

    rising_rsi = rsi(rising, length=2)
    falling_rsi = rsi(falling, length=2)

    assert (rising_rsi.dropna() == 100.0).all()
    assert (falling_rsi.dropna() == 0.0).all()
    # The first ``length`` rows stay missing until the smoothing window is full.
    assert rising_rsi.iloc[:2].isna().all()
    assert rising_rsi.notna().sum() == 3


def test_rsi_leaves_flat_series_missing() -> None:
    flat = pd.Series([50.0, 50.0, 50.0, 50.0], name="adjusted_close")

    result = rsi(flat, length=2)

    assert result.isna().all()


def test_rsi_stays_within_bounds() -> None:
    values = pd.Series([10.0, 11.0, 9.5, 12.0, 8.0, 13.0, 11.5, 14.0], name="adjusted_close")

    result = rsi(values, length=3).dropna()

    assert ((result >= 0.0) & (result <= 100.0)).all()


def test_rsi_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    result = rsi(values, length=1)

    pd.testing.assert_index_equal(result.index, values.index)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_rsi_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        rsi(
            _prices()["adjusted_close"],
            length=length,  # type: ignore[arg-type]
        )


def test_rsi_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = rsi(close, length=2)

    pd.testing.assert_index_equal(result.index, close.index)
    # AAA.ST rises monotonically, so its warmed-up RSI is a pure 100.
    aaa_rsi = result.loc[("yfinance", "AAA.ST")]
    assert (aaa_rsi.dropna() == 100.0).all()
    assert aaa_rsi.isna().sum() == 2
    # BBB.ST is flat and isolated, so it has no gains or losses.
    assert result.loc[("yfinance", "BBB.ST")].isna().all()


def test_rsi_returns_expected_values_for_mixed_price_changes() -> None:
    values = pd.Series(
        [10.0, 11.0, 9.5, 12.0, 8.0, 13.0, 11.5, 14.0],
        name="adjusted_close",
    )

    result = rsi(values, length=3)

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            79.3103448275862,
            35.3846153846154,
            68.3018867924528,
            55.5640832853026,
            69.6937969374878,
        ],
        name="rsi",
    )

    pd.testing.assert_series_equal(result, expected, check_exact=False)


def test_add_momentum_features_adds_stochastic_from_high_low_close() -> None:
    prices = _indexed_prices()

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
        stochastic_k_length=2,
        stochastic_k_smoothing=1,
        stochastic_d_length=1,
    )

    expected = stochastic_oscillator(
        prices[["high", "low", "close"]], k_length=2, k_smoothing=1, d_length=1
    )
    pd.testing.assert_series_equal(result["stochastic_k"], expected["stochastic_k"])
    pd.testing.assert_series_equal(result["stochastic_d"], expected["stochastic_d"])
    # AAA.ST rises steadily with the close sitting three-quarters up each two-day
    # high/low range, so its warmed-up %K is a constant 75.
    aaa_k = result.loc[("yfinance", "AAA.ST"), "stochastic_k"]
    assert (aaa_k.dropna() == 75.0).all()
    # BBB.ST is flat, so every window has no range and the oscillator stays
    # missing.
    assert result.loc[("yfinance", "BBB.ST"), "stochastic_k"].isna().all()


def test_add_momentum_features_delegates_to_stochastic_oscillator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _indexed_prices()
    calls = 0

    def fake_stochastic_oscillator(
        data: pd.DataFrame,
        *,
        k_length: int = 14,
        k_smoothing: int = 3,
        d_length: int = 3,
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        assert k_length == 2
        assert k_smoothing == 1
        assert d_length == 1
        assert list(data.columns) == ["high", "low", "close"]
        return pd.DataFrame(
            {
                "stochastic_k": [0.0] * len(data),
                "stochastic_d": [0.0] * len(data),
            },
            index=data.index,
        )

    monkeypatch.setattr(momentum_module, "stochastic_oscillator", fake_stochastic_oscillator)

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
        stochastic_k_length=2,
        stochastic_k_smoothing=1,
        stochastic_d_length=1,
    )

    # ``stochastic_oscillator`` already isolates provider/ticker groups
    # internally, so the orchestrator delegates to it once.
    assert calls == 1
    assert list(result.loc[:, ["stochastic_k", "stochastic_d"]].columns) == [
        "stochastic_k",
        "stochastic_d",
    ]


@pytest.mark.parametrize(
    ("k_length", "k_smoothing", "d_length"),
    [(0, 3, 3), (14, 0, 3), (14, 3, 0), (True, 3, 3), (14, 3, 1.5)],
)
def test_add_momentum_features_rejects_invalid_stochastic_lengths(
    k_length: object, k_smoothing: object, d_length: object
) -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="positive integer"):
        add_momentum_features(
            prices,
            ppo_lengths=(2, 3, 2),
            stochastic_k_length=k_length,  # type: ignore[arg-type]
            stochastic_k_smoothing=k_smoothing,  # type: ignore[arg-type]
            stochastic_d_length=d_length,  # type: ignore[arg-type]
        )


def test_add_momentum_features_requires_high_low_close() -> None:
    prices = _indexed_prices().drop(columns="high")

    with pytest.raises(ValueError, match="Missing required columns"):
        add_momentum_features(prices, ppo_lengths=(2, 3, 2))


def test_stochastic_oscillator_returns_dataframe_with_expected_columns_and_values() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 14.0, 13.0],
            "low": [9.0, 10.0, 8.0, 11.0, 7.0],
            "close": [10.0, 11.5, 9.0, 13.0, 8.0],
        }
    )

    result = stochastic_oscillator(frame, k_length=3, k_smoothing=1, d_length=2)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["stochastic_k", "stochastic_d"]
    pd.testing.assert_index_equal(result.index, frame.index)
    pd.testing.assert_series_equal(
        result["stochastic_k"],
        pd.Series(
            [
                np.nan,
                np.nan,
                20.0,
                100 * 5 / 6,
                100 * 1 / 7,
            ],
            name="stochastic_k",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["stochastic_d"],
        result["stochastic_k"].rolling(window=2, min_periods=2).mean().rename("stochastic_d"),
        check_exact=False,
    )


def test_stochastic_oscillator_smooths_slow_k_with_k_smoothing() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 14.0, 13.0],
            "low": [9.0, 10.0, 8.0, 11.0, 7.0],
            "close": [10.0, 11.5, 9.0, 13.0, 8.0],
        }
    )

    fast = stochastic_oscillator(frame, k_length=3, k_smoothing=1, d_length=1)
    slow = stochastic_oscillator(frame, k_length=3, k_smoothing=2, d_length=1)

    # The slow %K is the fast (raw) %K smoothed with a two-row simple average.
    pd.testing.assert_series_equal(
        slow["stochastic_k"],
        fast["stochastic_k"].rolling(window=2, min_periods=2).mean().rename("stochastic_k"),
        check_exact=False,
    )


def test_stochastic_oscillator_is_100_at_range_high_and_0_at_range_low() -> None:
    # The close sits at the highest high every row, so %K tops out at 100.
    top = pd.DataFrame(
        {
            "high": [2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 1.0, 1.0, 1.0],
            "close": [2.0, 3.0, 4.0, 5.0],
        }
    )
    # The close sits at the lowest low every row, so %K bottoms out at 0.
    bottom = pd.DataFrame(
        {
            "high": [5.0, 5.0, 5.0, 5.0],
            "low": [4.0, 3.0, 2.0, 1.0],
            "close": [4.0, 3.0, 2.0, 1.0],
        }
    )

    top_k = stochastic_oscillator(top, k_length=2, k_smoothing=1, d_length=1)["stochastic_k"]
    bottom_k = stochastic_oscillator(bottom, k_length=2, k_smoothing=1, d_length=1)["stochastic_k"]

    assert (top_k.dropna() == 100.0).all()
    assert (bottom_k.dropna() == 0.0).all()
    # The first ``k_length - 1`` rows stay missing until the window is full.
    assert top_k.iloc[:1].isna().all()
    assert top_k.notna().sum() == 3


def test_stochastic_oscillator_leaves_flat_window_missing() -> None:
    flat = pd.DataFrame(
        {
            "high": [50.0, 50.0, 50.0, 50.0],
            "low": [50.0, 50.0, 50.0, 50.0],
            "close": [50.0, 50.0, 50.0, 50.0],
        }
    )

    result = stochastic_oscillator(flat, k_length=2, k_smoothing=1, d_length=1)

    assert result["stochastic_k"].isna().all()
    assert result["stochastic_d"].isna().all()


def test_stochastic_oscillator_stays_within_bounds() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 13.0, 11.0, 14.0, 12.0, 15.0, 13.0, 16.0],
            "low": [8.0, 9.0, 6.0, 8.0, 5.0, 7.0, 4.0, 6.0],
            "close": [9.0, 12.0, 7.0, 13.0, 6.0, 14.0, 5.0, 15.0],
        }
    )

    result = stochastic_oscillator(frame, k_length=3)

    stochastic_k = result["stochastic_k"].dropna()
    stochastic_d = result["stochastic_d"].dropna()
    assert ((stochastic_k >= 0.0) & (stochastic_k <= 100.0)).all()
    assert ((stochastic_d >= 0.0) & (stochastic_d <= 100.0)).all()


def test_stochastic_oscillator_allows_non_temporal_index_and_preserves_row_order() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 15.0, 13.0],
            "low": [9.0, 13.0, 11.0],
            "close": [10.0, 14.0, 12.0],
        },
        index=pd.Index([2, 0, 1]),
    )

    result = stochastic_oscillator(frame, k_length=1, k_smoothing=1, d_length=1)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_stochastic_oscillator_requires_high_low_close() -> None:
    frame = _ohlc().drop(columns="close")

    with pytest.raises(ValueError, match="Missing required columns"):
        stochastic_oscillator(frame, k_length=2)


@pytest.mark.parametrize(
    ("k_length", "k_smoothing", "d_length"),
    [(0, 3, 3), (14, 0, 3), (14, 3, 0), (True, 3, 3), (14, 3, 1.5)],
)
def test_stochastic_oscillator_rejects_invalid_lengths(
    k_length: object, k_smoothing: object, d_length: object
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        stochastic_oscillator(
            _ohlc(),
            k_length=k_length,  # type: ignore[arg-type]
            k_smoothing=k_smoothing,  # type: ignore[arg-type]
            d_length=d_length,  # type: ignore[arg-type]
        )


def test_stochastic_oscillator_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = stochastic_oscillator(prices, k_length=2, k_smoothing=1, d_length=1)

    assert list(result.columns) == ["stochastic_k", "stochastic_d"]
    pd.testing.assert_index_equal(result.index, prices.index)
    # AAA.ST's close sits three-quarters up each two-day range, so its warmed-up
    # %K is a constant 75, isolated from BBB.ST.
    aaa_k = result.loc[("yfinance", "AAA.ST"), "stochastic_k"]
    assert (aaa_k.dropna() == 75.0).all()
    assert aaa_k.isna().sum() == 1
    # BBB.ST is flat and isolated, so every window has no range.
    assert result.loc[("yfinance", "BBB.ST"), "stochastic_k"].isna().all()


def test_add_momentum_features_adds_mfi_from_high_low_close_volume_and_mfi_percent_b() -> None:
    prices = _indexed_prices()

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
        mfi_length=2,
        mfi_bollinger_length=2,
    )

    expected = mfi(prices[["high", "low", "close", "volume"]], length=2)
    pd.testing.assert_series_equal(result["mfi"], expected.rename("mfi"), check_exact=False)

    expected_percent_b = bollinger_percent_b(result["mfi"], length=2, num_std=2.0)
    pd.testing.assert_series_equal(
        result["mfi_percent_b"],
        expected_percent_b.rename("mfi_percent_b"),
        check_exact=False,
    )
    # AAA.ST's typical price rises every day, so once the window warms up its MFI
    # is a pure 100.
    aaa_mfi = result.loc[("yfinance", "AAA.ST"), "mfi"]
    assert (aaa_mfi.dropna() == 100.0).all()
    # BBB.ST is flat, so its typical price never changes and MFI stays missing.
    assert result.loc[("yfinance", "BBB.ST"), "mfi"].isna().all()


def test_add_momentum_features_delegates_to_mfi(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _indexed_prices()
    calls = 0

    def fake_mfi(data: pd.DataFrame, *, length: int = 14) -> pd.Series:
        nonlocal calls
        calls += 1
        assert length == 2
        assert list(data.columns) == ["high", "low", "close", "volume"]
        return pd.Series([0.0] * len(data), index=data.index, name="mfi")

    monkeypatch.setattr(momentum_module, "mfi", fake_mfi)

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
        mfi_length=2,
    )

    # ``mfi`` already isolates provider/ticker groups internally, so the
    # orchestrator delegates to it once.
    assert calls == 1
    assert list(result.loc[:, ["mfi", "mfi_percent_b"]].columns) == ["mfi", "mfi_percent_b"]


def test_add_momentum_features_rejects_invalid_mfi_length() -> None:
    prices = _indexed_prices()

    with pytest.raises(ValueError, match="positive integer"):
        add_momentum_features(prices, ppo_lengths=(2, 3, 2), mfi_length=0)


def test_add_momentum_features_requires_volume() -> None:
    prices = _indexed_prices().drop(columns="volume")

    with pytest.raises(ValueError, match="Missing required columns"):
        add_momentum_features(prices, ppo_lengths=(2, 3, 2))


def test_mfi_returns_expected_values_for_mixed_price_changes() -> None:
    # high == low == close makes the typical price equal to the close, so the
    # money flow is driven purely by the hand-checked price changes below.
    frame = pd.DataFrame(
        {
            "high": [10.0, 11.0, 9.5, 12.0, 8.0],
            "low": [10.0, 11.0, 9.5, 12.0, 8.0],
            "close": [10.0, 11.0, 9.5, 12.0, 8.0],
            "volume": [100, 100, 100, 100, 100],
        }
    )

    result = mfi(frame, length=2)

    expected = pd.Series(
        [
            np.nan,
            np.nan,
            100 * 1100 / 2050,
            100 * 1200 / 2150,
            60.0,
        ],
        name="mfi",
    )

    pd.testing.assert_series_equal(result, expected, check_exact=False)


def test_mfi_is_100_without_negative_flow_and_0_without_positive_flow() -> None:
    rising = pd.DataFrame(
        {
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [1.0, 2.0, 3.0, 4.0, 5.0],
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [100, 100, 100, 100, 100],
        }
    )
    falling = pd.DataFrame(
        {
            "high": [5.0, 4.0, 3.0, 2.0, 1.0],
            "low": [5.0, 4.0, 3.0, 2.0, 1.0],
            "close": [5.0, 4.0, 3.0, 2.0, 1.0],
            "volume": [100, 100, 100, 100, 100],
        }
    )

    rising_mfi = mfi(rising, length=2)
    falling_mfi = mfi(falling, length=2)

    assert (rising_mfi.dropna() == 100.0).all()
    assert (falling_mfi.dropna() == 0.0).all()
    # The first ``length`` rows stay missing until the trailing window is full.
    assert rising_mfi.iloc[:2].isna().all()
    assert rising_mfi.notna().sum() == 3


def test_mfi_leaves_flat_window_missing() -> None:
    flat = pd.DataFrame(
        {
            "high": [50.0, 50.0, 50.0, 50.0],
            "low": [50.0, 50.0, 50.0, 50.0],
            "close": [50.0, 50.0, 50.0, 50.0],
            "volume": [100, 100, 100, 100],
        }
    )

    result = mfi(flat, length=2)

    assert result.isna().all()


def test_mfi_stays_within_bounds() -> None:
    frame = pd.DataFrame(
        {
            "high": [10.0, 13.0, 11.0, 14.0, 12.0, 15.0, 13.0, 16.0],
            "low": [8.0, 9.0, 6.0, 8.0, 5.0, 7.0, 4.0, 6.0],
            "close": [9.0, 12.0, 7.0, 13.0, 6.0, 14.0, 5.0, 15.0],
            "volume": [100, 250, 80, 300, 120, 400, 90, 500],
        }
    )

    result = mfi(frame, length=3).dropna()

    assert ((result >= 0.0) & (result <= 100.0)).all()


def test_mfi_allows_non_temporal_index_and_preserves_row_order() -> None:
    frame = pd.DataFrame(
        {
            "high": [11.0, 15.0, 13.0],
            "low": [9.0, 13.0, 11.0],
            "close": [10.0, 14.0, 12.0],
            "volume": [100, 100, 100],
        },
        index=pd.Index([2, 0, 1]),
    )

    result = mfi(frame, length=1)

    pd.testing.assert_index_equal(result.index, frame.index)


def test_mfi_requires_high_low_close_volume() -> None:
    frame = _ohlc().drop(columns="volume")

    with pytest.raises(ValueError, match="Missing required columns"):
        mfi(frame, length=2)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_mfi_rejects_invalid_length(length: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        mfi(
            _ohlc(),
            length=length,  # type: ignore[arg-type]
        )


def test_mfi_groups_by_ticker_index_levels() -> None:
    prices = _indexed_prices()

    result = mfi(prices, length=2)

    pd.testing.assert_index_equal(result.index, prices.index)
    # AAA.ST's typical price rises every day, so its warmed-up MFI is a pure 100,
    # isolated from BBB.ST.
    aaa_mfi = result.loc[("yfinance", "AAA.ST")]
    assert (aaa_mfi.dropna() == 100.0).all()
    assert aaa_mfi.isna().sum() == 2
    # BBB.ST is flat and isolated, so its typical price never changes.
    assert result.loc[("yfinance", "BBB.ST")].isna().all()


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
    expected_duration = consecutive_true_count(squeeze_on).rename("squeeze_duration")
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


def test_add_momentum_features_populates_and_scopes_squeeze_columns() -> None:
    prices = _squeeze_multi(include_adjusted_and_volume=True)

    result = add_momentum_features(
        prices,
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
        squeeze_bb_length=5,
        squeeze_kc_length=5,
        squeeze_atr_length=5,
    )

    # The raw price-unit line and the internal ATR/True Range scaffolding never
    # leak into the persisted feature frame.
    assert "squeeze_momentum" not in result.columns
    assert "atr" not in result.columns
    assert "true_range" not in result.columns
    assert result["squeeze_momentum_atr"].notna().any()
    assert result["squeeze_on"].notna().any()

    expected_squeeze = lazybear_squeeze_momentum(
        prices.loc[:, ["high", "low", "close"]],
        bb_length=5,
        kc_length=5,
        atr_length=5,
    ).drop(columns="squeeze_momentum")
    pd.testing.assert_frame_equal(result[expected_squeeze.columns], expected_squeeze)


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
    squeeze_duration = consecutive_true_count(squeeze_on)
    squeeze_release_duration = squeeze_duration.shift(1).where(squeeze_released)

    highest_high = high.rolling(window=kc_length, min_periods=kc_length).max()
    lowest_low = low.rolling(window=kc_length, min_periods=kc_length).min()
    range_midpoint = (highest_high + lowest_low) / 2.0
    reference_level = (range_midpoint + kc_basis) / 2.0
    momentum = linreg(close - reference_level, length=kc_length, offset=0)
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


def _multi_ticker_close() -> pd.Series:
    return _prices().set_index(["provider", "ticker", "trading_date"])["adjusted_close"]


def _ohlc() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"]).loc[("yfinance", "AAA.ST")]


def _indexed_prices() -> pd.DataFrame:
    return _prices().set_index(["provider", "ticker", "trading_date"])


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider": ["yfinance"] * 8,
            "ticker": [
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "AAA.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
                "BBB.ST",
            ],
            "trading_date": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-06",
                ]
            ).date,
            "high": [11.0, 13.0, 15.0, 17.0, 100.0, 100.0, 100.0, 100.0],
            "low": [9.0, 11.0, 13.0, 15.0, 100.0, 100.0, 100.0, 100.0],
            "close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
            "adjusted_close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1100, 1200, 1300, 500, 500, 500, 500],
        }
    )
