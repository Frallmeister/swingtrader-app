"""Tests for completed Zig Zag leg dynamics."""

import math

import pytest

from swingtrader.indicators.market_structure import (
    _zigzag_leg_dynamics,
    _ZigZagPivot,
)


@pytest.mark.parametrize(
    ("log_returns", "expected"),
    [
        ([0.20, -0.05, 0.20, -0.05, 0.20, -0.05], 0.60),
        ([0.05, -0.20, 0.05, -0.20, 0.05, -0.20], -0.60),
        ([0.10, -0.10, 0.10, -0.10, 0.10, -0.10], 0.0),
    ],
)
def test_zigzag_leg_dynamics_measure_balance_and_efficiency(
    log_returns: list[float],
    expected: float,
) -> None:
    pivots = _pivots_from_log_returns(log_returns)

    leg_balance, efficiency = _zigzag_leg_dynamics(pivots, leg_count=6)

    assert leg_balance == pytest.approx(expected)
    assert efficiency == pytest.approx(expected)


def test_zigzag_leg_balance_uses_median_magnitudes() -> None:
    pivots = _pivots_from_log_returns([0.10, -0.05, 1.00, -0.05, 0.10, -0.05])

    leg_balance, _ = _zigzag_leg_dynamics(pivots, leg_count=6)

    assert leg_balance == pytest.approx(1.0 / 3.0)


def test_zigzag_efficiency_uses_signed_net_log_displacement() -> None:
    log_returns = [0.20, -0.03, 0.10, -0.02, 0.15, -0.05]
    pivots = _pivots_from_log_returns(log_returns)

    _, efficiency = _zigzag_leg_dynamics(pivots, leg_count=6)

    expected = sum(log_returns) / sum(abs(value) for value in log_returns)
    assert efficiency == pytest.approx(expected)


def test_zigzag_leg_dynamics_wait_for_the_complete_window() -> None:
    pivots = _pivots_from_log_returns([0.10, -0.05, 0.10, -0.05, 0.10])

    leg_balance, efficiency = _zigzag_leg_dynamics(pivots, leg_count=6)

    assert math.isnan(leg_balance)
    assert math.isnan(efficiency)


@pytest.mark.parametrize(
    "prices",
    [
        [100.0] * 7,
        [100.0, 110.0, 0.0, 120.0, 110.0, 130.0, 120.0],
    ],
)
def test_zigzag_leg_dynamics_return_missing_for_invalid_path(
    prices: list[float],
) -> None:
    pivots = [
        _ZigZagPivot(
            position=position,
            price=price,
            direction=-1 if position % 2 == 0 else 1,
        )
        for position, price in enumerate(prices)
    ]

    leg_balance, efficiency = _zigzag_leg_dynamics(pivots, leg_count=6)

    assert math.isnan(leg_balance)
    assert math.isnan(efficiency)


def _pivots_from_log_returns(log_returns: list[float]) -> list[_ZigZagPivot]:
    price = 100.0
    pivots = [_ZigZagPivot(position=0, price=price, direction=-1)]

    for position, log_return in enumerate(log_returns, start=1):
        price *= math.exp(log_return)
        pivots.append(
            _ZigZagPivot(
                position=position,
                price=price,
                direction=1 if position % 2 == 1 else -1,
            )
        )

    return pivots
