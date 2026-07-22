"""Integration tests for point-in-time market-structure dynamics."""

import math

import pandas as pd
import pytest

from swingtrader.data.features.market_structure import (
    add_market_structure_features,
    zigzag_features,
)


def test_dynamics_appear_when_the_last_required_leg_is_confirmed() -> None:
    prices = _instrument_prices(
        [0.20, -0.05, 0.20, -0.05, 0.20, -0.05],
        ticker="AAA.ST",
    )

    result = _dynamics_features(prices).reset_index(drop=True)

    assert result.loc[:13, "market_structure_leg_balance"].isna().all()
    assert result.loc[:13, "market_structure_efficiency"].isna().all()
    assert result.loc[14, "market_structure_leg_balance"] == pytest.approx(0.60)
    assert result.loc[14, "market_structure_efficiency"] == pytest.approx(0.60)


def test_dynamics_exclude_close_movement_after_the_latest_pivot() -> None:
    prices = _instrument_prices(
        [0.20, -0.05, 0.20, -0.05, 0.20, -0.05],
        ticker="AAA.ST",
    )
    changed_close = prices.copy()
    changed_close.iloc[-1, changed_close.columns.get_loc("close")] *= 1.50

    original = _dynamics_features(prices).iloc[-1]
    changed = _dynamics_features(changed_close).iloc[-1]

    assert changed["market_structure_leg_balance"] == pytest.approx(
        original["market_structure_leg_balance"]
    )
    assert changed["market_structure_efficiency"] == pytest.approx(
        original["market_structure_efficiency"]
    )


def test_dynamics_are_append_invariant() -> None:
    prices = _instrument_prices(
        [0.20, -0.05, 0.20, -0.05, 0.20, -0.05, 0.15, -0.04],
        ticker="AAA.ST",
    )
    full_result = _dynamics_features(prices)

    for stop in range(1, len(prices) + 1):
        prefix_result = _dynamics_features(prices.iloc[:stop])
        pd.testing.assert_series_equal(
            prefix_result.iloc[-1],
            full_result.iloc[stop - 1],
            check_names=False,
        )


def test_dynamics_update_only_when_endpoint_replacement_is_confirmed() -> None:
    candidates = [100.0, 120.0, 108.0, 125.0, 112.0, 130.0, 117.0, 125.0, 110.0]
    prices = _prices_from_pivots(candidates, ticker="AAA.ST")
    result = zigzag_features(
        prices,
        deviation=9.9,
        pivot_legs=2,
        consistency_pivots=2,
        dynamics_legs=6,
    ).reset_index(drop=True)

    initial_balance, initial_efficiency = _expected_dynamics(candidates[:7])
    replaced_balance, replaced_efficiency = _expected_dynamics(
        [*candidates[:6], candidates[-1]]
    )

    assert result.loc[14, "market_structure_leg_balance"] == pytest.approx(
        initial_balance
    )
    assert result.loc[14, "market_structure_efficiency"] == pytest.approx(
        initial_efficiency
    )
    assert result.loc[17, "market_structure_leg_balance"] == pytest.approx(
        initial_balance
    )
    assert result.loc[17, "market_structure_efficiency"] == pytest.approx(
        initial_efficiency
    )
    assert result.loc[18, "market_structure_leg_balance"] == pytest.approx(
        replaced_balance
    )
    assert result.loc[18, "market_structure_efficiency"] == pytest.approx(
        replaced_efficiency
    )


def test_dynamics_are_scale_invariant_and_isolated_by_ticker() -> None:
    aaa = _instrument_prices(
        [0.20, -0.05, 0.20, -0.05, 0.20, -0.05],
        ticker="AAA.ST",
    )
    bbb = _scaled_copy(aaa, ticker="BBB.ST", scale=7.0)
    result = _dynamics_features(pd.concat([aaa, bbb]).sort_index())

    aaa_result = result.loc[("yfinance", "AAA.ST")]
    bbb_result = result.loc[("yfinance", "BBB.ST")]
    pd.testing.assert_frame_equal(
        aaa_result.reset_index(drop=True),
        bbb_result.reset_index(drop=True),
    )


def test_add_market_structure_features_forwards_dynamics_legs() -> None:
    prices = _instrument_prices(
        [0.20, -0.05, 0.20, -0.05],
        ticker="AAA.ST",
    )

    result = add_market_structure_features(
        prices,
        zigzag_deviation=0.0,
        zigzag_pivot_legs=2,
        zigzag_consistency_pivots=2,
        zigzag_dynamics_legs=4,
    )
    expected = zigzag_features(
        prices,
        deviation=0.0,
        pivot_legs=2,
        consistency_pivots=2,
        dynamics_legs=4,
    )

    pd.testing.assert_frame_equal(result[expected.columns], expected)


@pytest.mark.parametrize("dynamics_legs", [True, 1, 3, 4.5])
def test_dynamics_legs_must_be_an_even_integer_at_least_two(
    dynamics_legs: object,
) -> None:
    prices = _instrument_prices([0.10, -0.05], ticker="AAA.ST")

    with pytest.raises(ValueError, match="dynamics_legs must be an even integer"):
        zigzag_features(
            prices,
            deviation=0.0,
            pivot_legs=2,
            consistency_pivots=2,
            dynamics_legs=dynamics_legs,  # type: ignore[arg-type]
        )


def test_new_dynamics_columns_are_protected_from_overwrite() -> None:
    prices = _instrument_prices([0.10, -0.05], ticker="AAA.ST")
    prices["market_structure_efficiency"] = math.nan

    with pytest.raises(
        ValueError,
        match="Generated columns already exist.*market_structure_efficiency",
    ):
        add_market_structure_features(prices, zigzag_pivot_legs=2)


def _dynamics_features(prices: pd.DataFrame) -> pd.DataFrame:
    return zigzag_features(
        prices,
        deviation=0.0,
        pivot_legs=2,
        consistency_pivots=2,
        dynamics_legs=6,
    )


def _instrument_prices(
    log_returns: list[float],
    *,
    ticker: str,
) -> pd.DataFrame:
    pivots = [100.0]
    for log_return in log_returns:
        pivots.append(pivots[-1] * math.exp(log_return))
    return _prices_from_pivots(pivots, ticker=ticker)


def _prices_from_pivots(pivots: list[float], *, ticker: str) -> pd.DataFrame:
    prices = [(pivots[0] + pivots[1]) / 2.0]
    for current, following in zip(pivots, pivots[1:], strict=False):
        prices.extend([current, (current + following) / 2.0])
    prices.extend([pivots[-1], (pivots[-2] + pivots[-1]) / 2.0])

    dates = pd.date_range("2026-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "high": prices,
            "low": prices,
            "close": prices,
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["yfinance"] * len(prices),
                [ticker] * len(prices),
                dates,
            ],
            names=["provider", "ticker", "trading_date"],
        ),
    )


def _expected_dynamics(pivots: list[float]) -> tuple[float, float]:
    log_returns = [
        math.log(current / previous)
        for previous, current in zip(pivots, pivots[1:], strict=False)
    ]
    upward = sorted(abs(value) for value in log_returns[::2])
    downward = sorted(abs(value) for value in log_returns[1::2])
    upward_median = upward[len(upward) // 2]
    downward_median = downward[len(downward) // 2]
    balance = (upward_median - downward_median) / (upward_median + downward_median)
    efficiency = sum(log_returns) / sum(abs(value) for value in log_returns)
    return balance, efficiency


def _scaled_copy(
    prices: pd.DataFrame,
    *,
    ticker: str,
    scale: float,
) -> pd.DataFrame:
    result = prices.mul(scale)
    result.index = pd.MultiIndex.from_arrays(
        [
            result.index.get_level_values("provider"),
            [ticker] * len(result),
            result.index.get_level_values("trading_date"),
        ],
        names=result.index.names,
    )
    return result
