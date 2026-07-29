from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select

from swingtrader.modeling.labeling import (
    EMA_COLORS,
    LABEL_TABLE_NAME,
    SELECTED_TRACE_NAME,
    STOP_TRACE_NAME,
    TAKE_PROFIT_TRACE_NAME,
    LabelingConfig,
    add_pivot_annotation,
    build_labeling_figure,
    calculate_forward_outcomes,
    candle_labels,
    create_labeling_session,
    load_labeling_session,
    load_labels,
    load_latest_labeling_session,
    plan_labeling_windows,
    prepare_adjustment_consistent_prices,
    prepare_chart_view,
    prepare_labeling_frame,
    risk_guide_for_date,
    save_labeling_window,
    slice_chart_context,
    update_risk_guide_traces,
    update_selected_trace,
)


def _prices(periods: int = 140) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=periods)
    close = pd.Series(np.linspace(100.0, 130.0, periods), index=index)
    open_ = close - np.where(np.arange(periods) % 2 == 0, 0.4, -0.4)
    return pd.DataFrame(
        {
            "open": open_,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adjusted_close": close,
            "volume": np.arange(periods) + 1_000,
        },
        index=index,
    )


def test_labeling_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="window_size"):
        LabelingConfig(window_size=0)
    with pytest.raises(ValueError, match="step_size"):
        LabelingConfig(window_size=10, step_size=11)
    with pytest.raises(ValueError, match="commission_rate"):
        LabelingConfig(commission_rate=1.0)
    with pytest.raises(ValueError, match="default_heatmap_mode"):
        LabelingConfig(default_heatmap_mode="annualized")  # type: ignore[arg-type]


def test_plan_windows_uses_fixed_step_and_reserves_future_rows() -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    config = LabelingConfig(window_size=10, step_size=7, forward_horizon=3)

    windows = plan_labeling_windows(
        dates,
        config=config,
        labeling_start_date=dates[2],
        labeling_end_date=dates[-1],
    )

    assert [window.start_position for window in windows] == [2, 9, 16, 23]
    assert [window.end_position for window in windows] == [11, 18, 25, 32]
    assert all(len(window.trading_dates) == 10 for window in windows)
    assert windows[-1].end_position + config.forward_horizon < len(dates)


def test_forward_outcomes_are_commission_aware_in_all_modes() -> None:
    index = pd.bdate_range("2020-01-01", periods=3)
    close = pd.Series([100.0, 110.0, 121.0], index=index)
    atr_values = pd.Series([5.0, 5.0, 5.0], index=index)
    config = LabelingConfig(
        window_size=1,
        step_size=1,
        forward_horizon=2,
        atr_stop_multiple=2.0,
        commission_rate=0.0025,
    )

    outcomes = calculate_forward_outcomes(close, atr_values, config=config)

    expected_return = 110.0 * 0.9975 / (100.0 * 1.0025) - 1.0
    expected_pnl = 110.0 * 0.9975 - 100.0 * 1.0025
    assert outcomes.net_return.loc[1, index[0]] == pytest.approx(expected_return)
    assert outcomes.atr_units.loc[1, index[0]] == pytest.approx(expected_pnl / 5.0)
    assert outcomes.risk_units.loc[1, index[0]] == pytest.approx(expected_pnl / 10.0)
    assert np.isnan(outcomes.net_return.loc[2, index[1]])


def test_adjustment_consistent_prices_removes_placeholders_and_scales_ohlc() -> None:
    prices = _prices(periods=6)
    placeholder_date = prices.index[2]
    prices.loc[placeholder_date, ["open", "high", "low", "close"]] = prices.loc[
        prices.index[1], ["open", "high", "low", "close"]
    ]
    prices.loc[placeholder_date, "adjusted_close"] = prices.loc[prices.index[1], "adjusted_close"]
    prices.loc[placeholder_date, "volume"] = 0
    prices.loc[prices.index[3] :, "adjusted_close"] *= 0.5

    adjusted = prepare_adjustment_consistent_prices(prices)

    assert placeholder_date not in adjusted.index
    assert adjusted.loc[prices.index[3], "close"] == pytest.approx(
        prices.loc[prices.index[3], "adjusted_close"]
    )
    assert adjusted.loc[prices.index[3], "high"] == pytest.approx(
        prices.loc[prices.index[3], "high"] * 0.5
    )


def test_prepare_frame_and_chart_use_requested_visual_contract() -> None:
    config = LabelingConfig(window_size=40, step_size=30, forward_horizon=5)
    frame = prepare_labeling_frame(_prices(), config=config)
    windows = plan_labeling_windows(
        frame.index,
        config=config,
        labeling_end_date=frame.index[-1],
    )
    context = slice_chart_context(frame, window=windows[1], config=config)
    selected_date = windows[1].trading_dates[5]

    figure = build_labeling_figure(
        context,
        window=windows[1],
        selected_dates={selected_date},
        config=config,
    )

    colors = {trace.name: trace.line.color for trace in figure.data if trace.name.startswith("EMA")}
    assert colors == {
        "EMA 10": EMA_COLORS[10],
        "EMA 20": EMA_COLORS[20],
        "EMA 50": EMA_COLORS[50],
    }
    assert {STOP_TRACE_NAME, TAKE_PROFIT_TRACE_NAME, SELECTED_TRACE_NAME} <= {
        trace.name for trace in figure.data
    }
    volume = next(trace for trace in figure.data if trace.name == "Volume")
    assert set(volume.marker.color) == {"#26a69a", "#ef5350"}
    heatmap = next(trace for trace in figure.data if trace.name == "Forward outcomes")
    assert list(heatmap.y) == [1, 2, 3, 4, 5]
    assert figure.layout.xaxis.range[0] == windows[1].start_date
    assert figure.layout.xaxis.range[1] == windows[1].end_date
    assert len(figure.layout.shapes) == 6
    assert all(shape.fillcolor == "#6b7280" for shape in figure.layout.shapes)
    assert all(shape.opacity == pytest.approx(0.12) for shape in figure.layout.shapes)

    update_selected_trace(figure, context, set())
    selected_trace = next(trace for trace in figure.data if trace.name == SELECTED_TRACE_NAME)
    assert list(selected_trace.x) == []

    guide = risk_guide_for_date(context, trading_date=selected_date, config=config)
    assert guide is not None
    update_risk_guide_traces(figure, guide)
    stop_trace = next(trace for trace in figure.data if trace.name == STOP_TRACE_NAME)
    target_trace = next(trace for trace in figure.data if trace.name == TAKE_PROFIT_TRACE_NAME)
    assert stop_trace.visible is True
    assert target_trace.visible is True
    assert stop_trace.y[0] == pytest.approx(guide.stop)
    assert target_trace.y[0] == pytest.approx(guide.take_profit)


def test_prepare_chart_view_centres_window_for_wide_timeframe() -> None:
    config = LabelingConfig(window_size=40, step_size=30, forward_horizon=5)
    frame = prepare_labeling_frame(_prices(periods=1000), config=config)
    windows = plan_labeling_windows(
        frame.index,
        config=config,
        labeling_end_date=frame.index[-1],
    )
    window = windows[len(windows) // 2]
    window_centre = (window.start_position + window.end_position) // 2

    default_context, default_view = prepare_chart_view(frame, window=window, config=config)
    default_start = int(frame.index.get_loc(default_view[0]))
    default_end = int(frame.index.get_loc(default_view[1]))
    assert default_end - default_start == 84
    assert (default_start + default_end) // 2 == window_centre

    context, (view_start, view_end) = prepare_chart_view(
        frame, window=window, config=config, timeframe="3Y"
    )
    view_start_pos = int(frame.index.get_loc(view_start))
    view_end_pos = int(frame.index.get_loc(view_end))

    # the window centre sits exactly in the middle of a 756-session span
    assert view_end_pos - view_start_pos == 756
    assert (view_start_pos + view_end_pos) // 2 == window_centre
    assert view_end_pos - view_start_pos > window.end_position - window.start_position
    # the context slice contains the whole visible span
    assert context.index[0] <= view_start
    assert view_end <= context.index[-1]


def test_prepare_chart_view_keeps_window_centred_at_data_edge() -> None:
    config = LabelingConfig(window_size=40, step_size=30, forward_horizon=5)
    frame = prepare_labeling_frame(_prices(periods=200), config=config)
    windows = plan_labeling_windows(
        frame.index,
        config=config,
        labeling_end_date=frame.index[-1],
    )
    window = windows[0]

    context, (view_start, view_end) = prepare_chart_view(
        frame, window=window, config=config, timeframe="3Y"
    )

    # The first window cannot reach 1.5 years of real history on the left, so
    # the viewport extrapolates business days before the data to stay centred.
    assert view_start < frame.index[0]
    window_centre_date = frame.index[(window.start_position + window.end_position) // 2]
    left_span = window_centre_date - view_start
    right_span = view_end - window_centre_date
    assert abs((left_span - right_span).days) <= 3
    # the context slice never reaches beyond the loaded data
    assert context.index[0] >= frame.index[0]
    assert context.index[-1] <= frame.index[-1]


def test_prepare_chart_view_rejects_unknown_timeframe() -> None:
    config = LabelingConfig(window_size=40, step_size=30, forward_horizon=5)
    frame = prepare_labeling_frame(_prices(), config=config)
    window = plan_labeling_windows(
        frame.index,
        config=config,
        labeling_end_date=frame.index[-1],
    )[0]
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        prepare_chart_view(frame, window=window, config=config, timeframe="5Y")


def test_build_labeling_figure_opens_on_supplied_viewport() -> None:
    config = LabelingConfig(window_size=40, step_size=30, forward_horizon=5)
    frame = prepare_labeling_frame(_prices(periods=500), config=config)
    windows = plan_labeling_windows(
        frame.index,
        config=config,
        labeling_end_date=frame.index[-1],
    )
    window = windows[len(windows) // 2]
    context, viewport = prepare_chart_view(frame, window=window, config=config, timeframe="1Y")

    figure = build_labeling_figure(
        context,
        window=window,
        selected_dates=set(),
        config=config,
        viewport_range=viewport,
    )

    assert figure.layout.xaxis.range[0] == viewport[0]
    assert figure.layout.xaxis.range[1] == viewport[1]
    assert figure.layout.yaxis.range is not None


def test_pivot_annotation_preserves_established_appearance() -> None:
    config = LabelingConfig(window_size=40, step_size=30, forward_horizon=5)
    frame = prepare_labeling_frame(_prices(), config=config)
    window = plan_labeling_windows(
        frame.index,
        config=config,
        labeling_end_date=frame.index[-1],
    )[0]
    context = slice_chart_context(frame, window=window, config=config)
    figure = build_labeling_figure(
        context,
        window=window,
        selected_dates=set(),
        config=config,
    )

    add_pivot_annotation(
        figure,
        x=window.start_date,
        y=100.0,
        kind="low",
    )

    annotation = figure.layout.annotations[-1]
    assert annotation.text == "100.0"
    assert annotation.font.family == "Courier New, monospace"
    assert annotation.font.size == 10
    assert annotation.font.color == "#39da89"
    assert annotation.arrowcolor == "#636363"
    assert annotation.bgcolor == "#484848"
    assert annotation.opacity == pytest.approx(0.8)
    assert annotation.ay > 0


def test_persistence_saves_full_window_resumes_and_updates_overlap() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    config = LabelingConfig(window_size=5, step_size=3, forward_horizon=1)
    dates = pd.bdate_range("2020-01-01", periods=12)
    windows = plan_labeling_windows(
        dates,
        config=config,
        labeling_end_date=dates[-1],
    )
    session = create_labeling_session(
        engine=engine,
        provider="yfinance",
        tickers=("AAA.ST", "BBB.ST"),
        label_family="trend_continuation",
        labeling_start_date=date(2020, 1, 1),
        labeling_end_date=dates[-1],
        config=config,
        labeling_session_id="session-1",
    )

    updated = save_labeling_window(
        engine=engine,
        labeling_session_id=session.labeling_session_id,
        ticker="AAA.ST",
        window=windows[0],
        positive_dates={windows[0].trading_dates[1]},
        config=config.with_calibration(atr_stop_multiple=1.5, reward_risk_ratio=2.5),
        next_ticker_position=0,
        next_window_position=1,
    )

    first_labels = load_labels(
        engine=engine,
        provider="yfinance",
        ticker="AAA.ST",
        start_date=windows[0].start_date,
        end_date=windows[0].end_date,
    )
    assert len(first_labels) == 5
    assert sum(first_labels.values()) == 1
    assert updated.current_window_position == 1
    assert updated.config.atr_stop_multiple == 1.5
    assert updated.config.reward_risk_ratio == 2.5
    assert load_latest_labeling_session(engine=engine) == updated
    assert (
        load_latest_labeling_session(
            engine=engine,
            provider="other-provider",
            label_family="trend_continuation",
        )
        is None
    )

    overlap_date = windows[1].trading_dates[0]
    save_labeling_window(
        engine=engine,
        labeling_session_id=session.labeling_session_id,
        ticker="AAA.ST",
        window=windows[1],
        positive_dates={overlap_date},
        config=updated.config,
    )

    with engine.connect() as connection:
        row_count = connection.scalar(select(func.count()).select_from(candle_labels))
        table_name = candle_labels.name
    assert table_name == LABEL_TABLE_NAME
    assert row_count == 8  # 5 rows plus 3 new rows in the overlapping window.
    overlap_label = load_labels(
        engine=engine,
        provider="yfinance",
        ticker="AAA.ST",
        start_date=overlap_date,
        end_date=overlap_date,
    )
    assert overlap_label == {overlap_date: 1}

    loaded = load_labeling_session(engine=engine, labeling_session_id="session-1")
    assert loaded.current_ticker == "AAA.ST"


def test_unique_key_excludes_label_family_and_later_save_replaces_it() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    config = LabelingConfig(window_size=2, step_size=1, forward_horizon=1)
    dates = pd.bdate_range("2020-01-01", periods=4)
    window = plan_labeling_windows(
        dates,
        config=config,
        labeling_end_date=dates[-1],
    )[0]
    first = create_labeling_session(
        engine=engine,
        provider="yfinance",
        tickers=("AAA.ST",),
        label_family="trend_continuation",
        labeling_end_date=dates[-1],
        config=config,
        labeling_session_id="first",
    )
    save_labeling_window(
        engine=engine,
        labeling_session_id=first.labeling_session_id,
        ticker="AAA.ST",
        window=window,
        positive_dates={window.trading_dates[0]},
        config=config,
        completed=True,
    )
    second = create_labeling_session(
        engine=engine,
        provider="yfinance",
        tickers=("AAA.ST",),
        label_family="breakout",
        labeling_end_date=dates[-1],
        config=config,
        labeling_session_id="second",
    )
    save_labeling_window(
        engine=engine,
        labeling_session_id=second.labeling_session_id,
        ticker="AAA.ST",
        window=window,
        positive_dates=set(),
        config=config,
    )

    with engine.connect() as connection:
        rows = connection.execute(select(candle_labels)).mappings().all()
    assert len(rows) == 2
    assert {row["label_family"] for row in rows} == {"breakout"}
    assert {row["label"] for row in rows} == {0}


def test_save_rejects_stale_session_window() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    config = LabelingConfig(window_size=2, step_size=1, forward_horizon=1)
    dates = pd.bdate_range("2020-01-01", periods=5)
    windows = plan_labeling_windows(
        dates,
        config=config,
        labeling_end_date=dates[-1],
    )
    session = create_labeling_session(
        engine=engine,
        provider="yfinance",
        tickers=("AAA.ST",),
        label_family="trend_continuation",
        labeling_end_date=dates[-1],
        config=config,
    )

    with pytest.raises(ValueError, match="currently points to window 0"):
        save_labeling_window(
            engine=engine,
            labeling_session_id=session.labeling_session_id,
            ticker="AAA.ST",
            window=windows[1],
            positive_dates=set(),
            config=config,
        )


def test_save_rejects_changes_to_fixed_session_configuration() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    config = LabelingConfig(window_size=2, step_size=1, forward_horizon=1)
    dates = pd.bdate_range("2020-01-01", periods=4)
    window = plan_labeling_windows(
        dates,
        config=config,
        labeling_end_date=dates[-1],
    )[0]
    session = create_labeling_session(
        engine=engine,
        provider="yfinance",
        tickers=("AAA.ST",),
        label_family="trend_continuation",
        labeling_end_date=dates[-1],
        config=config,
    )

    with pytest.raises(ValueError, match="fixed fields changed: commission_rate"):
        save_labeling_window(
            engine=engine,
            labeling_session_id=session.labeling_session_id,
            ticker="AAA.ST",
            window=window,
            positive_dates={window.trading_dates[0]},
            config=LabelingConfig(
                window_size=2,
                step_size=1,
                forward_horizon=1,
                commission_rate=0.001,
            ),
        )

    assert (
        load_labels(
            engine=engine,
            provider="yfinance",
            ticker="AAA.ST",
            start_date=window.start_date,
            end_date=window.end_date,
        )
        == {}
    )


def test_save_rejects_selected_date_outside_planned_window() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    config = LabelingConfig(window_size=2, step_size=1, forward_horizon=1)
    dates = pd.bdate_range("2020-01-01", periods=4)
    window = plan_labeling_windows(
        dates,
        config=config,
        labeling_end_date=dates[-1],
    )[0]
    session = create_labeling_session(
        engine=engine,
        provider="yfinance",
        tickers=("AAA.ST",),
        label_family="trend_continuation",
        labeling_end_date=dates[-1],
        config=config,
    )

    with pytest.raises(ValueError, match="outside the planned window"):
        save_labeling_window(
            engine=engine,
            labeling_session_id=session.labeling_session_id,
            ticker="AAA.ST",
            window=window,
            positive_dates={dates[-1]},
            config=config,
        )
