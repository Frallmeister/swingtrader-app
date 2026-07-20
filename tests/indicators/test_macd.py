import numpy as np
import pandas as pd
import pytest

from swingtrader.indicators.macd import macd, ppo


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
        }
    )
