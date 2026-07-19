import numpy as np
import pandas as pd
import pytest

import swingtrader.data.features.momentum as momentum_module
from swingtrader.data.features.momentum import add_momentum_features, macd, ppo, ppo_percentile


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


def _multi_ticker_close() -> pd.Series:
    return _prices().set_index(["provider", "ticker", "trading_date"])["adjusted_close"]


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
            "adjusted_close": [10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
