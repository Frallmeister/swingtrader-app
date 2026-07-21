import numpy as np
import pandas as pd

from swingtrader.data.features import (
    add_market_structure_features,
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
    market_structure_columns = set(add_market_structure_features(data).columns)
    momentum_columns = set(add_momentum_features(data).columns)
    volatility_columns = set(add_volatility_features(data).columns)
    expected_columns = (
        returns_columns
        | trend_columns
        | momentum_columns
        | volatility_columns
        | market_structure_columns
    )

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
    manual = (
        data.pipe(add_return_features)
        .pipe(add_trend_features)
        .pipe(add_momentum_features)
        .pipe(add_volatility_features)
        .pipe(add_market_structure_features)
    )

    pd.testing.assert_frame_equal(result, manual, check_exact=False)


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 140
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
