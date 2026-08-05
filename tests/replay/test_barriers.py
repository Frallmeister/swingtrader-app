import pytest

from swingtrader.replay.barriers import BarrierExit, resolve_long_barriers
from swingtrader.replay.domain import BarrierPolicy, DailyBar


def test_open_gap_is_resolved_before_intraday_policy():
    result = resolve_long_barriers(
        DailyBar(
            trading_date=__import__("datetime").date(2020, 1, 2),
            open=90,
            high=110,
            low=89,
            close=105,
        ),
        stop_price=95,
        target_price=108,
        policy=BarrierPolicy.TARGET_FIRST,
    )
    assert result is not None
    assert result.exit is BarrierExit.STOP
    assert result.execution_price == 90


@pytest.mark.parametrize(
    ("policy", "close", "expected"),
    [
        (BarrierPolicy.STOP_FIRST, 101, BarrierExit.STOP),
        (BarrierPolicy.TARGET_FIRST, 101, BarrierExit.TARGET),
        (BarrierPolicy.CANDLE_PATH, 99, BarrierExit.TARGET),
        (BarrierPolicy.CANDLE_PATH, 101, BarrierExit.STOP),
    ],
)
def test_ambiguous_barrier_policy(policy, close, expected):
    from datetime import date

    result = resolve_long_barriers(
        DailyBar(date(2020, 1, 2), open=100, high=110, low=90, close=close),
        stop_price=95,
        target_price=105,
        policy=policy,
    )
    assert result is not None
    assert result.exit is expected
