import numpy as np
import pandas as pd
import pytest

import swingtrader.data.features.momentum as momentum_module
from swingtrader.data.features.momentum import add_momentum_features, ppo_percentile
from swingtrader.indicators.macd import ppo
from swingtrader.indicators.oscillators import rsi, stochastic_oscillator
from swingtrader.indicators.squeeze_momentum import lazybear_squeeze_momentum
from swingtrader.indicators.volatility import bollinger_percent_b
from swingtrader.indicators.volume import mfi


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
