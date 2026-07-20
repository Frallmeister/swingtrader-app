import numpy as np
import pandas as pd
import pytest

from swingtrader.data.features import (
    add_momentum_features,
    add_return_features,
    add_trend_features,
    add_volatility_features,
)
from swingtrader.data.features.pipeline import add_default_features


def test_add_default_features_includes_all_family_columns_without_duplicates() -> None:
    data = _prices()

    result = add_default_features(data)

    returns_columns = set(add_return_features(data).columns)
    trend_columns = set(add_trend_features(data).columns)
    momentum_columns = set(add_momentum_features(data).columns)
    volatility_columns = set(add_volatility_features(data).columns)
    expected_columns = returns_columns | trend_columns | momentum_columns | volatility_columns

    assert set(result.columns) == expected_columns
    assert result.columns.is_unique


def test_add_default_features_preserves_index_and_row_order() -> None:
    data = _prices()

    result = add_default_features(data)

    pd.testing.assert_index_equal(result.index, data.index)


def test_add_default_features_does_not_mutate_input() -> None:
    data = _prices()
    original = data.copy(deep=True)

    add_default_features(data)

    pd.testing.assert_frame_equal(data, original)


def test_add_default_features_matches_manual_family_chain() -> None:
    data = _prices()

    result = add_default_features(data)
    manual = add_volatility_features(
        add_momentum_features(add_trend_features(add_return_features(data)))
    )

    pd.testing.assert_frame_equal(result, manual, check_exact=False)


def test_default_features_preserve_pre_refactor_outputs() -> None:
    prices = _regression_prices()

    result = add_default_features(prices)

    expected_columns = [
        # Exact pre-refactor column sequence.
    ]
    assert list(result.columns) == expected_columns

    expected = {
        ("yfinance", "AAA.ST", pd.Timestamp("2026-06-30")): {
            "ema_fast_to_ema_mid": ...,
            "adx": ...,
            "ppo": ...,
            "ppo_percentile": ...,
            "rsi": ...,
            "squeeze_momentum_atr": ...,
            "atr_percent": ...,
            "bollinger_percent_b": ...,
        },
        ("yfinance", "BBB.ST", pd.Timestamp("2026-06-30")): {
            # Captured pre-refactor values.
        },
    }

    for index, columns in expected.items():
        for column, value in columns.items():
            assert result.loc[index, column] == pytest.approx(
                value,
                rel=1e-12,
                abs=1e-12,
            )


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 60
    trading_dates = [
        timestamp.date() for timestamp in pd.date_range("2026-01-01", periods=n, freq="B")
    ]

    frames = []
    for ticker, base in (("AAA.ST", 100.0), ("BBB.ST", 50.0)):
        steps = rng.normal(0.0, 1.0, n)
        close = base + np.cumsum(steps)
        span = np.abs(rng.normal(0.0, 1.0, n)) + 0.5
        frame = pd.DataFrame(
            {
                "high": close + span,
                "low": close - span,
                "close": close,
                "adjusted_close": close,
                "volume": rng.integers(1_000, 5_000, n).astype(float),
            }
        )
        frame.index = pd.MultiIndex.from_arrays(
            [["yfinance"] * n, [ticker] * n, trading_dates],
            names=["provider", "ticker", "trading_date"],
        )
        frames.append(frame)
    return pd.concat(frames).sort_index()


def _regression_prices() -> pd.DataFrame:
    n = 160
    trading_dates = [
        timestamp.date()
        for timestamp in pd.date_range("2025-01-01", periods=n, freq="B")
    ]
    step = np.arange(n, dtype=float)

    frames = []

    for ticker, base, slope, phase in (
        ("AAA.ST", 100.0, 0.20, 0.0),
        ("BBB.ST", 500.0, -0.15, 1.5),
    ):
        close = (
            base
            + slope * step
            + 3.0 * np.sin(step / 6.0 + phase)
            + 0.5 * np.cos(step / 2.5)
        )
        span = 1.5 + 0.4 * np.sin(step / 8.0 + phase) ** 2

        frame = pd.DataFrame(
            {
                "high": close + span,
                "low": close - span * 0.9,
                "close": close,
                "adjusted_close": close * 0.98,
                "volume": 100_000.0 + 500.0 * step + 10_000.0 * np.cos(
                    step / 9.0 + phase
                ),
            }
        )
        frame.index = pd.MultiIndex.from_arrays(
            [
                ["yfinance"] * n,
                [ticker] * n,
                trading_dates,
            ],
            names=["provider", "ticker", "trading_date"],
        )
        frames.append(frame)

    return pd.concat(frames).sort_index()
