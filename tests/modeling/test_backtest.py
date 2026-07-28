from __future__ import annotations

import pandas as pd
import pytest

from swingtrader.modeling.backtest import run_backtest, summarize_backtest


def _prices(rows: list[tuple[str, str, str, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=("provider", "ticker", "trading_date", "open", "high", "low", "close"),
    )
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    return frame.set_index(["provider", "ticker", "trading_date"]).sort_index()


def _signals(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=("provider", "ticker", "trading_date", "score", "atr"),
    )
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    return frame.set_index(["provider", "ticker", "trading_date"]).sort_index()


def test_next_open_entry_uses_atr_risk_sizing_and_fixed_two_r_target() -> None:
    prices = _prices(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 100, 105, 95, 100),
            ("yfinance", "AAA.ST", "2026-01-05", 100, 121, 99, 115),
            ("yfinance", "AAA.ST", "2026-01-06", 116, 118, 112, 117),
        ]
    )
    signals = _signals([("yfinance", "AAA.ST", "2026-01-02", 0.9, 10.0)])

    result = run_backtest(
        prices,
        signals,
        initial_cash=10_000,
        risk_fraction=0.01,
        max_positions=1,
        max_holding_sessions=5,
        reward_risk_ratio=2.0,
        commission_rate=0.0,
    )

    trade = result["trades"].iloc[0]
    assert trade["entry_date"] == pd.Timestamp("2026-01-05")
    assert trade["quantity"] == 10
    assert trade["stop_price"] == pytest.approx(90.0)
    assert trade["take_profit_price"] == pytest.approx(120.0)
    assert trade["exit_price"] == pytest.approx(120.0)
    assert trade["reward_risk"] == pytest.approx(2.0)
    assert trade["holding_sessions"] == 1
    assert result["summary"]["win_rate"] == pytest.approx(1.0)
    assert result["summary"]["total_return"] == pytest.approx(0.02)


def test_same_bar_stop_and_target_uses_conservative_stop_first_policy() -> None:
    prices = _prices(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 100, 105, 95, 100),
            ("yfinance", "AAA.ST", "2026-01-05", 100, 121, 89, 110),
            ("yfinance", "AAA.ST", "2026-01-06", 110, 112, 108, 111),
        ]
    )
    prices["adjusted_close"] = [50.0, 55.0, 55.5]
    signals = _signals([("yfinance", "AAA.ST", "2026-01-02", 0.9, 10.0)])

    trade = run_backtest(
        prices,
        signals,
        initial_cash=10_000,
        risk_fraction=0.01,
        max_positions=1,
        max_holding_sessions=5,
        stop_atr_multiple=1.0,
        commission_rate=0.0,
    )["trades"].iloc[0]

    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == pytest.approx(90.0)
    assert bool(trade["ambiguous_intrabar"])
    assert trade["reward_risk"] == pytest.approx(-1.0)


def test_timeout_is_decided_after_close_and_executed_at_next_open() -> None:
    prices = _prices(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 100, 102, 98, 100),
            ("yfinance", "AAA.ST", "2026-01-05", 100, 105, 95, 102),
            ("yfinance", "AAA.ST", "2026-01-06", 103, 106, 99, 104),
            ("yfinance", "AAA.ST", "2026-01-07", 105, 107, 101, 106),
        ]
    )
    signals = _signals([("yfinance", "AAA.ST", "2026-01-02", 0.9, 10.0)])

    trade = run_backtest(
        prices,
        signals,
        initial_cash=10_000,
        risk_fraction=0.01,
        max_positions=1,
        max_holding_sessions=2,
        stop_atr_multiple=1.0,
        commission_rate=0.0,
    )["trades"].iloc[0]

    assert trade["entry_date"] == pd.Timestamp("2026-01-05")
    assert trade["exit_date"] == pd.Timestamp("2026-01-07")
    assert trade["exit_price"] == pytest.approx(105.0)
    assert trade["holding_sessions"] == 2
    assert trade["exit_reason"] == "timeout"


def test_timeout_exit_frees_a_slot_for_an_entry_at_the_same_next_open() -> None:
    prices = _prices(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 100, 102, 98, 100),
            ("yfinance", "AAA.ST", "2026-01-05", 100, 104, 96, 101),
            ("yfinance", "AAA.ST", "2026-01-06", 102, 104, 98, 103),
            ("yfinance", "AAA.ST", "2026-01-07", 103, 105, 100, 104),
            ("yfinance", "BBB.ST", "2026-01-02", 50, 52, 48, 50),
            ("yfinance", "BBB.ST", "2026-01-05", 50, 52, 48, 51),
            ("yfinance", "BBB.ST", "2026-01-06", 52, 56, 50, 55),
            ("yfinance", "BBB.ST", "2026-01-07", 55, 56, 53, 54),
        ]
    )
    signals = _signals(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 0.9, 10.0),
            ("yfinance", "BBB.ST", "2026-01-05", 0.8, 5.0),
        ]
    )

    trades = run_backtest(
        prices,
        signals,
        initial_cash=10_000,
        risk_fraction=0.01,
        max_positions=1,
        max_holding_sessions=1,
        stop_atr_multiple=1.0,
        commission_rate=0.0,
    )["trades"]

    aaa = trades.loc[trades["ticker"].eq("AAA.ST")].iloc[0]
    bbb = trades.loc[trades["ticker"].eq("BBB.ST")].iloc[0]
    assert aaa["exit_date"] == pd.Timestamp("2026-01-06")
    assert bbb["entry_date"] == pd.Timestamp("2026-01-06")


def test_daily_ranking_selects_only_the_highest_score_when_one_slot_is_available() -> None:
    prices = _prices(
        [
            ("yfinance", ticker, trading_date, 100, 105, 95, 100)
            for ticker in ("AAA.ST", "BBB.ST")
            for trading_date in ("2026-01-02", "2026-01-05", "2026-01-06")
        ]
    )
    signals = _signals(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 0.4, 10.0),
            ("yfinance", "BBB.ST", "2026-01-02", 0.9, 10.0),
        ]
    )

    trades = run_backtest(
        prices,
        signals,
        initial_cash=10_000,
        risk_fraction=0.01,
        max_positions=1,
        max_holding_sessions=1,
        stop_atr_multiple=1.0,
        commission_rate=0.0,
    )["trades"]

    assert trades["ticker"].tolist() == ["BBB.ST"]


def test_summary_reports_requested_commission_and_reward_risk_metrics() -> None:
    trades = pd.DataFrame(
        {
            "net_pnl": [200.0, -100.0],
            "reward_risk": [2.0, -1.0],
            "holding_sessions": [3, 5],
            "entry_commission": [10.0, 10.0],
            "exit_commission": [12.0, 9.0],
        }
    )

    summary = summarize_backtest(trades, initial_cash=10_000, final_equity=10_100)

    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["average_reward_risk_win"] == pytest.approx(2.0)
    assert summary["average_reward_risk_loss"] == pytest.approx(-1.0)
    assert summary["total_reward_risk"] == pytest.approx(1.0)
    assert summary["total_return"] == pytest.approx(0.01)
    assert summary["expectancy"] == pytest.approx(0.5)
    assert summary["average_holding_sessions"] == pytest.approx(4.0)
    assert summary["commissions_paid"] == pytest.approx(41.0)


def test_signals_must_match_raw_price_rows() -> None:
    prices = _prices([("yfinance", "AAA.ST", "2026-01-02", 100, 105, 95, 100)])
    signals = _signals([("yfinance", "BBB.ST", "2026-01-02", 0.9, 10.0)])

    with pytest.raises(ValueError, match="Every signal row"):
        run_backtest(
            prices,
            signals,
            initial_cash=10_000,
            risk_fraction=0.01,
            max_positions=1,
            max_holding_sessions=5,
        )


def test_opening_gap_exits_at_the_raw_open_price() -> None:
    prices = _prices(
        [
            ("yfinance", "AAA.ST", "2026-01-02", 100, 102, 98, 100),
            ("yfinance", "AAA.ST", "2026-01-05", 100, 105, 95, 102),
            ("yfinance", "AAA.ST", "2026-01-06", 85, 90, 80, 88),
            ("yfinance", "AAA.ST", "2026-01-07", 88, 90, 86, 89),
        ]
    )
    signals = _signals([("yfinance", "AAA.ST", "2026-01-02", 0.9, 10.0)])

    trade = run_backtest(
        prices,
        signals,
        initial_cash=10_000,
        risk_fraction=0.01,
        max_positions=1,
        max_holding_sessions=5,
        stop_atr_multiple=1.0,
        commission_rate=0.0,
    )["trades"].iloc[0]

    assert trade["stop_price"] == pytest.approx(90.0)
    assert trade["exit_date"] == pd.Timestamp("2026-01-06")
    assert trade["exit_price"] == pytest.approx(85.0)
    assert trade["reward_risk"] == pytest.approx(-1.5)
