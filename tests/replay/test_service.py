from datetime import date

import numpy as np
import pytest

from swingtrader.replay.domain import BarrierPolicy, CourtageProfileName, PositionDecisionAction
from swingtrader.replay.metrics import calculate_performance_metrics
from swingtrader.replay.service import InsufficientCashError, ReplayValidationError


def create_replay(service, *, cash=50_000):
    return service.create_replay(
        name="Test replay",
        provider="yfinance",
        tickers=["AAA.ST", "BBB.ST"],
        start_date=date(2020, 1, 2),
        end_date=date(2020, 1, 9),
        initial_cash=cash,
        courtage_profile=CourtageProfileName.MINI,
        barrier_policy=BarrierPolicy.CANDLE_PATH,
    )


def test_evening_morning_buy_and_resume(service):
    state = create_replay(service)
    replay_id = state["session"]["id"]
    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.BUY,
        ticker="AAA.ST",
        allocation_sek=10_000,
        stop_price=90,
        target_price=130,
    )
    morning = service.finalize_evening(replay_id)
    assert morning["session"]["phase"] == "morning"
    assert morning["morning_open_prices"]["AAA.ST"] == pytest.approx(101)

    evening = service.complete_morning(replay_id)
    assert evening["session"]["phase"] == "evening"
    position = next(item for item in evening["positions"] if item["status"] == "open")
    assert position["ticker"] == "AAA.ST"
    assert position["quantity"] > 0
    assert evening["session"]["cash"] < 50_000

    resumed = service.get_state(replay_id)
    assert resumed["positions"] == evening["positions"]


def test_evening_transition_requires_decision_for_every_position(service):
    state = create_replay(service)
    replay_id = state["session"]["id"]
    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.BUY,
        ticker="AAA.ST",
        quantity=10,
        stop_price=90,
        target_price=130,
    )
    service.finalize_evening(replay_id)
    service.complete_morning(replay_id)
    with pytest.raises(ReplayValidationError, match="explicit decision"):
        service.finalize_evening(replay_id)


def test_insufficient_cash_rejects_purchase_without_fill(service):
    state = create_replay(service, cash=100)
    replay_id = state["session"]["id"]
    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.BUY,
        ticker="AAA.ST",
        quantity=1,
    )
    service.finalize_evening(replay_id)
    with pytest.raises(InsufficientCashError):
        service.complete_morning(replay_id)
    assert service.repository.fills(replay_id) == []
    assert service.get_state(replay_id)["session"]["phase"] == "morning"


def test_morning_chart_hides_current_high_low_close(service):
    state = create_replay(service)
    replay_id = state["session"]["id"]
    service.finalize_evening(replay_id)
    chart = service.chart(replay_id, "AAA.ST", [], lookback_sessions=30)
    assert chart["current_open"] == pytest.approx(101)
    assert chart["bars"][-1]["time"] == "2020-01-02"


def test_partial_reduction_charges_a_separate_sell_courtage(service):
    state = create_replay(service)
    replay_id = state["session"]["id"]
    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.BUY,
        ticker="AAA.ST",
        quantity=10,
        stop_price=90,
        target_price=130,
    )
    service.finalize_evening(replay_id)
    state = service.complete_morning(replay_id)
    position = next(item for item in state["positions"] if item["status"] == "open")

    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.REDUCE,
        ticker="AAA.ST",
        position_id=position["id"],
        quantity=4,
        stop_price=91,
        target_price=131,
    )
    service.finalize_evening(replay_id)
    state = service.complete_morning(replay_id)

    remaining = next(item for item in state["positions"] if item["id"] == position["id"])
    assert remaining["quantity"] == 6
    fills = service.repository.fills(replay_id)
    assert [fill["side"] for fill in fills] == ["buy", "sell"]
    assert fills[-1]["courtage"] == pytest.approx(1.04)


def test_invalid_morning_buy_is_rejected_before_a_planned_sale_is_written(service):
    state = create_replay(service)
    replay_id = state["session"]["id"]
    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.BUY,
        ticker="AAA.ST",
        quantity=10,
        stop_price=90,
        target_price=130,
    )
    service.finalize_evening(replay_id)
    state = service.complete_morning(replay_id)
    position = next(item for item in state["positions"] if item["status"] == "open")

    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.SELL,
        ticker="AAA.ST",
        position_id=position["id"],
    )
    service.save_evening_decision(
        replay_id,
        action=PositionDecisionAction.BUY,
        ticker="BBB.ST",
        quantity=1,
        stop_price=999,
    )
    service.finalize_evening(replay_id)

    with pytest.raises(ReplayValidationError, match="below its entry price"):
        service.complete_morning(replay_id)
    assert len(service.repository.fills(replay_id)) == 1
    assert service.repository.get_position(position["id"])["status"] == "open"


def test_watchlist_item_can_be_removed_and_added_again(service):
    state = create_replay(service)
    replay_id = state["session"]["id"]
    state = service.add_watchlist(replay_id, "AAA.ST")
    first_id = state["watchlist"][0]["id"]
    service.remove_watchlist(replay_id, first_id)
    state = service.add_watchlist(replay_id, "AAA.ST")
    assert len(state["watchlist"]) == 1
    assert state["watchlist"][0]["id"] != first_id


def test_state_exposes_fills_and_append_only_events(service):
    state = create_replay(service)
    assert state["fills"] == []
    assert state["events"][0]["event_type"] == "replay_created"


def test_sortino_uses_all_daily_returns_for_downside_deviation():
    metrics = calculate_performance_metrics(
        initial_cash=100.0,
        cash=98.0,
        positions=[],
        close_prices={},
        previous_snapshots=[{"equity": 100.0}, {"equity": 101.0}, {"equity": 99.0}],
    )
    returns = np.asarray([0.01, 99 / 101 - 1, 98 / 99 - 1])
    expected_downside = np.sqrt(np.mean(np.square(np.minimum(returns, 0.0))))
    expected = np.mean(returns) / expected_downside * np.sqrt(252.0)
    assert metrics["sortino_ratio"] == pytest.approx(expected)
