import numpy as np
import pandas as pd
import pytest

import swingtrader.data.features.trends as trend_module
from swingtrader.data.features.trends import add_trend_features, ema, ppo, ppo_percentile, sma


def test_add_trend_features_preserves_source_columns_and_adds_final_features() -> None:
    prices = _prices()
    original = prices.copy(deep=True)

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    expected_columns = [
        *prices.columns,
        "sma_fast_to_sma_slow",
        "ema_fast_to_ema_slow",
        "ema_fast_to_sma_fast",
        "ppo",
        "ppo_signal",
        "ppo_histogram",
        "ppo_percentile",
    ]
    assert list(result.columns) == expected_columns
    assert "sma_fast" not in result.columns
    assert "sma_slow" not in result.columns
    assert "ema_fast" not in result.columns
    assert "ema_slow" not in result.columns
    pd.testing.assert_index_equal(result.index, prices.index)
    pd.testing.assert_frame_equal(prices, original)

    pd.testing.assert_series_equal(
        result["sma_fast_to_sma_slow"],
        pd.Series(
            [np.nan, np.nan, 13.0 / 12.0 - 1.0, 15.0 / 14.0 - 1.0, np.nan, np.nan, 0.0, 0.0],
            name="sma_fast_to_sma_slow",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ppo"],
        result["ema_fast_to_ema_slow"].rename("ppo"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ema_fast_to_sma_fast"],
        pd.Series(
            [
                np.nan,
                11.333333333333332 / 11.0 - 1.0,
                13.11111111111111 / 13.0 - 1.0,
                15.037037037037036 / 15.0 - 1.0,
                np.nan,
                0.0,
                0.0,
                0.0,
            ],
            name="ema_fast_to_sma_fast",
        ),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ppo_histogram"],
        result["ppo"].sub(result["ppo_signal"]).rename("ppo_histogram"),
        check_exact=False,
    )
    pd.testing.assert_series_equal(
        result["ppo_percentile"],
        pd.Series(
            [np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, 1.0],
            name="ppo_percentile",
        ),
        check_exact=False,
    )
    assert "ppo" not in prices.columns


def test_add_trend_features_accepts_identifiers_as_index_levels() -> None:
    prices = _prices().set_index(["provider", "ticker", "trading_date"])

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
    )

    pd.testing.assert_index_equal(result.index, prices.index)
    assert result.loc[("yfinance", "AAA.ST"), "sma_fast_to_sma_slow"].notna().sum() == 2
    assert result.loc[("yfinance", "BBB.ST"), "sma_fast_to_sma_slow"].notna().sum() == 2


def test_add_trend_features_rejects_unordered_input() -> None:
    prices = _prices().iloc[[1, 0, 2, 3, 4, 5, 6, 7]].reset_index(drop=True)

    with pytest.raises(ValueError, match="strictly ordered"):
        add_trend_features(prices, fast_slow_lengths=(2, 3), ppo_lengths=(2, 3, 2))


def test_add_trend_features_calls_ppo_once_per_group(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = _prices()
    calls = 0

    def fake_ppo(
        values: pd.Series, *, lengths: tuple[int, int, int] = (12, 26, 9), use_percent: bool = True
    ) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        assert lengths == (2, 3, 2)
        return pd.DataFrame(
            {
                "ppo": [0.0] * len(values),
                "ppo_signal": [0.0] * len(values),
                "ppo_histogram": [0.0] * len(values),
            },
            index=values.index,
        )

    monkeypatch.setattr(trend_module, "ppo", fake_ppo)

    result = add_trend_features(
        prices,
        fast_slow_lengths=(2, 3),
        ppo_lengths=(2, 3, 2),
        ppo_percentile_min_history=1,
    )

    assert calls == 2
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


def test_sma_and_ema_calculate_one_sequence() -> None:
    prices = _prices()["adjusted_close"].iloc[:4]

    simple = sma(prices, length=2)
    exponential = ema(prices, length=2)

    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 11.0, 13.0, 15.0], name="adjusted_close"),
    )
    pd.testing.assert_series_equal(
        exponential,
        pd.Series(
            [
                np.nan,
                11.333333333333332,
                13.11111111111111,
                15.037037037037036,
            ],
            name="adjusted_close",
        ),
        check_exact=False,
    )


def test_sma_and_ema_allow_ordered_datetime_index() -> None:
    values = pd.Series(
        [10.0, 12.0, 14.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"]),
        name="adjusted_close",
    )

    simple = sma(values, length=2)
    exponential = ema(values, length=2)

    pd.testing.assert_index_equal(simple.index, values.index)
    pd.testing.assert_index_equal(exponential.index, values.index)
    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 11.0, 13.0], index=values.index, name="adjusted_close"),
    )


def test_sma_and_ema_reject_unordered_datetime_index() -> None:
    values = pd.Series(
        [10.0, 14.0, 12.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-03", "2026-07-02"]),
        name="adjusted_close",
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        sma(values, length=2)

    with pytest.raises(ValueError, match="chronologically ordered"):
        ema(values, length=2)


def test_sma_and_ema_allow_ordered_period_index() -> None:
    values = pd.Series(
        [10.0, 12.0, 14.0],
        index=pd.period_range("2026-07-01", periods=3, freq="D"),
        name="adjusted_close",
    )

    simple = sma(values, length=2)
    exponential = ema(values, length=2)

    pd.testing.assert_index_equal(simple.index, values.index)
    pd.testing.assert_index_equal(exponential.index, values.index)


def test_sma_and_ema_reject_unordered_trading_date_multiindex() -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance", "yfinance", "yfinance"],
            ["AAA.ST", "AAA.ST", "AAA.ST"],
            pd.to_datetime(["2026-07-01", "2026-07-03", "2026-07-02"]),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    values = pd.Series([10.0, 14.0, 12.0], index=index, name="adjusted_close")

    with pytest.raises(ValueError, match="chronologically ordered"):
        sma(values, length=2)

    with pytest.raises(ValueError, match="chronologically ordered"):
        ema(values, length=2)


def test_sma_and_ema_allow_non_temporal_index_and_preserve_row_order() -> None:
    values = pd.Series([10.0, 14.0, 12.0], index=pd.Index([2, 0, 1]), name="adjusted_close")

    simple = sma(values, length=2)
    exponential = ema(values, length=2)

    pd.testing.assert_index_equal(simple.index, values.index)
    pd.testing.assert_index_equal(exponential.index, values.index)
    pd.testing.assert_series_equal(
        simple,
        pd.Series([np.nan, 12.0, 13.0], index=values.index, name="adjusted_close"),
    )


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


def test_ppo_rejects_unordered_datetime_index() -> None:
    values = pd.Series(
        [10.0, 14.0, 12.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-03", "2026-07-02"]),
        name="adjusted_close",
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        ppo(values, lengths=(1, 2, 1))


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


def test_ppo_percentile_rejects_unordered_datetime_index() -> None:
    values = pd.Series(
        [1.0, 3.0, 2.0],
        index=pd.to_datetime(["2026-07-01", "2026-07-03", "2026-07-02"]),
        name="ppo",
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        ppo_percentile(values)


def test_ppo_percentile_allows_non_temporal_index_and_preserves_row_order() -> None:
    values = pd.Series([1.0, 3.0, 2.0], index=pd.Index([2, 0, 1]), name="ppo")

    result = ppo_percentile(values, min_history=1)

    pd.testing.assert_index_equal(result.index, values.index)


def test_trend_helpers_reject_invalid_inputs() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="fast length"):
        add_trend_features(prices, fast_slow_lengths=(3, 2), ppo_lengths=(2, 3, 2))

    with pytest.raises(ValueError, match="positive integer"):
        sma(prices["adjusted_close"], length=0)

    with pytest.raises(ValueError, match="Missing required columns"):
        add_trend_features(prices.drop(columns="adjusted_close"))

    with pytest.raises(ValueError, match="fast length"):
        ppo(prices["adjusted_close"], lengths=(3, 2, 1))

    with pytest.raises(ValueError, match="positive integer"):
        add_trend_features(prices, ppo_percentile_min_history=0)


@pytest.mark.parametrize("length", [0, -1, True, 2.5, "2"])
def test_sma_and_ema_reject_invalid_lengths(length: object) -> None:
    values = _prices()["adjusted_close"]

    with pytest.raises(ValueError, match="positive integer"):
        sma(
            values,
            length=length,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="positive integer"):
        ema(
            values,
            length=length,  # type: ignore[arg-type]
        )


def test_sma_groups_by_ticker_index_levels_without_cross_ticker_bleed() -> None:
    close = _multi_ticker_close()

    result = sma(close, length=2)

    pd.testing.assert_index_equal(result.index, close.index)
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "AAA.ST")].reset_index(drop=True),
        pd.Series([np.nan, 11.0, 13.0, 15.0], name="adjusted_close"),
    )
    # The first BBB.ST value is a warm-up NaN; if AAA.ST's tail bled across the
    # boundary this would instead be mean(16, 100) == 58.0.
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series([np.nan, 100.0, 100.0, 100.0], name="adjusted_close"),
    )


def test_ema_groups_by_ticker_index_levels_without_cross_ticker_bleed() -> None:
    close = _multi_ticker_close()

    result = ema(close, length=2)

    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price feeds an EMA of exactly that price after warm-up; any
    # cross-ticker state from AAA.ST would perturb these BBB.ST values.
    pd.testing.assert_series_equal(
        result.loc[("yfinance", "BBB.ST")].reset_index(drop=True),
        pd.Series([np.nan, 100.0, 100.0, 100.0], name="adjusted_close"),
    )


def test_ppo_groups_by_ticker_index_levels() -> None:
    close = _multi_ticker_close()

    result = ppo(close, lengths=(1, 2, 1))

    assert list(result.columns) == ["ppo", "ppo_signal", "ppo_histogram"]
    pd.testing.assert_index_equal(result.index, close.index)
    # A constant price yields a PPO of zero after warm-up, isolated from AAA.ST.
    bbb_ppo = result.loc[("yfinance", "BBB.ST"), "ppo"]
    assert (bbb_ppo.dropna() == 0.0).all()
    assert bbb_ppo.isna().sum() == 1


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


def test_primitives_preserve_interleaved_multi_ticker_row_order() -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * 6,
            ["AAA.ST", "BBB.ST", "AAA.ST", "BBB.ST", "AAA.ST", "BBB.ST"],
            pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-03",
                ]
            ),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    values = pd.Series([10.0, 100.0, 12.0, 100.0, 14.0, 100.0], index=index, name="adjusted_close")

    result = sma(values, length=2)

    pd.testing.assert_index_equal(result.index, values.index)
    pd.testing.assert_series_equal(
        result,
        pd.Series(
            [np.nan, np.nan, 11.0, 100.0, 13.0, 100.0],
            index=index,
            name="adjusted_close",
        ),
    )


@pytest.mark.parametrize(
    "indicator",
    [
        lambda values: sma(values, length=2),
        lambda values: ema(values, length=2),
        lambda values: ppo(values, lengths=(1, 2, 1)),
        lambda values: ppo_percentile(values, min_history=1),
    ],
)
def test_primitives_reject_unordered_dates_within_one_ticker(
    indicator: object,
) -> None:
    index = pd.MultiIndex.from_arrays(
        [
            ["yfinance"] * 6,
            ["AAA.ST", "AAA.ST", "AAA.ST", "BBB.ST", "BBB.ST", "BBB.ST"],
            pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                    "2026-07-01",
                    "2026-07-03",
                    "2026-07-02",
                ]
            ),
        ],
        names=["provider", "ticker", "trading_date"],
    )
    values = pd.Series([10.0, 11.0, 12.0, 100.0, 101.0, 102.0], index=index, name="adjusted_close")

    with pytest.raises(ValueError, match="chronologically ordered"):
        indicator(values)  # type: ignore[operator]


def _multi_ticker_close() -> pd.Series:
    return _prices().set_index(["provider", "ticker", "trading_date"])["adjusted_close"]


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
            "adjusted_close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
