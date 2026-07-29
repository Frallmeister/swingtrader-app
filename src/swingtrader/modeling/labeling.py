"""Reusable contracts and calculations for interactive candle labeling.

The module owns deterministic rolling windows, indicator-enriched chart data,
forward-outcome calculations, Plotly figure construction, and SQLite/PostgreSQL
persistence. Notebook code remains responsible for mutable widget state and event
callbacks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    select,
)
from sqlalchemy.engine import Connection, Engine

from swingtrader.indicators import atr, ema, pivot_points_high_low

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

HeatmapMode = Literal["net_return", "atr_units", "risk_units"]

LABEL_TABLE_NAME = "modeling_candle_labels"
SESSION_TABLE_NAME = "modeling_labeling_sessions"
SUPPORTED_HEATMAP_MODES: tuple[HeatmapMode, ...] = (
    "net_return",
    "atr_units",
    "risk_units",
)
EMA_COLORS = {
    10: "#1f77b4",
    20: "#ff7f0e",
    50: "#9467bd",
}
SELECTED_TRACE_NAME = "Selected entries"
STOP_TRACE_NAME = "ATR stop"
TAKE_PROFIT_TRACE_NAME = "Take profit"
PRICE_HUD_NAME = "price-hud"

metadata = MetaData()

candle_labels = Table(
    LABEL_TABLE_NAME,
    metadata,
    Column("provider", String(64), primary_key=True),
    Column("ticker", String(32), primary_key=True),
    Column("trading_date", Date, primary_key=True),
    Column("label_family", String(64), nullable=False),
    Column("label", Integer, nullable=False),
    Column("labeling_session_id", String(36), nullable=False),
    Column("labeled_at", DateTime(timezone=True), nullable=False),
    CheckConstraint("label IN (0, 1)", name="ck_modeling_candle_labels_binary"),
)

labeling_sessions = Table(
    SESSION_TABLE_NAME,
    metadata,
    Column("labeling_session_id", String(36), primary_key=True),
    Column("provider", String(64), nullable=False),
    Column("tickers_json", Text, nullable=False),
    Column("label_family", String(64), nullable=False),
    Column("labeling_start_date", Date, nullable=True),
    Column("labeling_end_date", Date, nullable=False),
    Column("current_ticker_position", Integer, nullable=False),
    Column("current_window_position", Integer, nullable=False),
    Column("window_size", Integer, nullable=False),
    Column("step_size", Integer, nullable=False),
    Column("forward_horizon", Integer, nullable=False),
    Column("atr_length", Integer, nullable=False),
    Column("atr_stop_multiple", Float, nullable=False),
    Column("reward_risk_ratio", Float, nullable=False),
    Column("commission_rate", Float, nullable=False),
    Column("pivot_high_left", Integer, nullable=False),
    Column("pivot_high_right", Integer, nullable=False),
    Column("pivot_low_left", Integer, nullable=False),
    Column("pivot_low_right", Integer, nullable=False),
    Column("default_heatmap_mode", String(32), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("completed", Boolean, nullable=False, default=False),
)


@dataclass(frozen=True, slots=True)
class LabelingConfig:
    """Configuration shared by one labeling session and its chart calculations."""

    window_size: int = 80
    step_size: int = 60
    forward_horizon: int = 10
    atr_length: int = 14
    atr_stop_multiple: float = 1.0
    reward_risk_ratio: float = 2.0
    commission_rate: float = 0.0025
    pivot_high_left: int = 15
    pivot_high_right: int = 15
    pivot_low_left: int = 15
    pivot_low_right: int = 15
    default_heatmap_mode: HeatmapMode = "net_return"

    def __post_init__(self) -> None:
        for name in (
            "window_size",
            "step_size",
            "forward_horizon",
            "atr_length",
            "pivot_high_left",
            "pivot_high_right",
            "pivot_low_left",
            "pivot_low_right",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}.")
        if self.step_size > self.window_size:
            raise ValueError("step_size must not exceed window_size.")
        if self.atr_stop_multiple <= 0:
            raise ValueError("atr_stop_multiple must be positive.")
        if self.reward_risk_ratio <= 0:
            raise ValueError("reward_risk_ratio must be positive.")
        if not 0 <= self.commission_rate < 1:
            raise ValueError("commission_rate must be in the interval [0, 1).")
        if self.default_heatmap_mode not in SUPPORTED_HEATMAP_MODES:
            raise ValueError(
                "default_heatmap_mode must be one of "
                f"{SUPPORTED_HEATMAP_MODES}; got {self.default_heatmap_mode!r}."
            )

    def with_calibration(
        self,
        *,
        atr_stop_multiple: float,
        reward_risk_ratio: float,
    ) -> LabelingConfig:
        """Return a validated copy with updated visual risk calibration."""

        return replace(
            self,
            atr_stop_multiple=atr_stop_multiple,
            reward_risk_ratio=reward_risk_ratio,
        )


@dataclass(frozen=True, slots=True)
class LabelingWindow:
    """One deterministic planned window over an instrument's trading sessions."""

    position: int
    start_position: int
    end_position: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    trading_dates: tuple[pd.Timestamp, ...]


@dataclass(frozen=True, slots=True)
class LabelingSession:
    """Persisted progress and configuration for a resumable labeling workflow."""

    labeling_session_id: str
    provider: str
    tickers: tuple[str, ...]
    label_family: str
    labeling_start_date: date | None
    labeling_end_date: date
    current_ticker_position: int
    current_window_position: int
    config: LabelingConfig
    created_at: datetime
    updated_at: datetime
    completed: bool

    @property
    def current_ticker(self) -> str:
        """Return the current ticker, rejecting completed or malformed sessions."""

        if self.completed:
            raise ValueError("The labeling session is complete.")
        if not 0 <= self.current_ticker_position < len(self.tickers):
            raise ValueError("The current ticker position is outside the session ticker list.")
        return self.tickers[self.current_ticker_position]


@dataclass(frozen=True, slots=True)
class ForwardOutcomeMatrices:
    """Commission-aware forward outcomes sharing horizons and entry dates."""

    net_return: pd.DataFrame
    atr_units: pd.DataFrame
    risk_units: pd.DataFrame

    def for_mode(self, mode: HeatmapMode) -> pd.DataFrame:
        """Return the matrix corresponding to ``mode``."""

        if mode not in SUPPORTED_HEATMAP_MODES:
            raise ValueError(f"Unsupported heatmap mode: {mode!r}.")
        return getattr(self, mode)


@dataclass(frozen=True, slots=True)
class RiskGuide:
    """Hover-driven close-based stop and take-profit guide."""

    trading_date: pd.Timestamp
    end_date: pd.Timestamp
    entry: float
    atr: float
    stop: float
    take_profit: float


def initialize_labeling_tables(engine: Engine) -> None:
    """Create the labeling and session tables when they do not already exist."""

    metadata.create_all(engine, tables=[candle_labels, labeling_sessions])


def create_labeling_session(
    *,
    engine: Engine,
    provider: str,
    tickers: Sequence[str],
    label_family: str,
    labeling_end_date: date | str | pd.Timestamp,
    config: LabelingConfig,
    labeling_start_date: date | str | pd.Timestamp | None = None,
    labeling_session_id: str | None = None,
) -> LabelingSession:
    """Create and persist a new labeling session at its first planned window.

    The ordered ticker list and fixed window configuration are stored with the
    session so a later notebook run can resume deterministically. Visual ATR and
    reward/risk calibration may be changed later when a window is saved.
    """

    normalized_tickers = _normalize_tickers(tickers)
    normalized_provider = _required_text(provider, name="provider")
    normalized_family = _required_text(label_family, name="label_family")
    session_id = labeling_session_id or str(uuid4())
    now = _utc_now()
    start_date = _to_date(labeling_start_date) if labeling_start_date is not None else None
    end_date = _to_date(labeling_end_date)
    if start_date is not None and start_date > end_date:
        raise ValueError("labeling_start_date must not be later than labeling_end_date.")

    values = {
        "labeling_session_id": session_id,
        "provider": normalized_provider,
        "tickers_json": json.dumps(normalized_tickers),
        "label_family": normalized_family,
        "labeling_start_date": start_date,
        "labeling_end_date": end_date,
        "current_ticker_position": 0,
        "current_window_position": 0,
        **_config_values(config),
        "created_at": now,
        "updated_at": now,
        "completed": False,
    }
    initialize_labeling_tables(engine)
    with engine.begin() as connection:
        connection.execute(labeling_sessions.insert().values(**values))
    return _session_from_mapping(values)


def load_labeling_session(*, engine: Engine, labeling_session_id: str) -> LabelingSession:
    """Load one persisted labeling session by identifier."""

    initialize_labeling_tables(engine)
    statement = select(labeling_sessions).where(
        labeling_sessions.c.labeling_session_id == labeling_session_id
    )
    with engine.connect() as connection:
        row = connection.execute(statement).mappings().one_or_none()
    if row is None:
        raise ValueError(f"Unknown labeling session: {labeling_session_id!r}.")
    return _session_from_mapping(row)


def load_latest_labeling_session(
    *,
    engine: Engine,
    provider: str | None = None,
    label_family: str | None = None,
    include_completed: bool = False,
) -> LabelingSession | None:
    """Load the most recently updated session matching the optional workflow filters."""

    initialize_labeling_tables(engine)
    statement = select(labeling_sessions)
    if provider is not None:
        statement = statement.where(labeling_sessions.c.provider == provider)
    if label_family is not None:
        statement = statement.where(labeling_sessions.c.label_family == label_family)
    if not include_completed:
        statement = statement.where(labeling_sessions.c.completed.is_(False))
    statement = statement.order_by(
        labeling_sessions.c.updated_at.desc(),
        labeling_sessions.c.created_at.desc(),
        labeling_sessions.c.labeling_session_id.desc(),
    ).limit(1)
    with engine.connect() as connection:
        row = connection.execute(statement).mappings().one_or_none()
    return None if row is None else _session_from_mapping(row)


def load_labels(
    *,
    engine: Engine,
    provider: str,
    ticker: str,
    start_date: date | str | pd.Timestamp,
    end_date: date | str | pd.Timestamp,
) -> dict[pd.Timestamp, int]:
    """Load authoritative binary labels for one instrument and inclusive date range."""

    initialize_labeling_tables(engine)
    statement = (
        select(candle_labels.c.trading_date, candle_labels.c.label)
        .where(
            and_(
                candle_labels.c.provider == provider,
                candle_labels.c.ticker == ticker,
                candle_labels.c.trading_date >= _to_date(start_date),
                candle_labels.c.trading_date <= _to_date(end_date),
            )
        )
        .order_by(candle_labels.c.trading_date)
    )
    with engine.connect() as connection:
        rows = connection.execute(statement).all()
    return {pd.Timestamp(trading_date): int(label) for trading_date, label in rows}


def save_labeling_window(
    *,
    engine: Engine,
    labeling_session_id: str,
    ticker: str,
    window: LabelingWindow,
    positive_dates: Iterable[date | str | pd.Timestamp],
    config: LabelingConfig,
    next_ticker_position: int | None = None,
    next_window_position: int | None = None,
    completed: bool | None = None,
) -> LabelingSession:
    """Persist a full inspected window and optional progress change atomically.

    Every planned-window candle is written: selected dates receive label ``1``
    and unselected dates receive label ``0``. Dates exposed only by Plotly panning
    are absent from ``window.trading_dates`` and therefore remain unlabeled.
    """

    initialize_labeling_tables(engine)
    positive = {_to_timestamp(value) for value in positive_dates}
    planned = set(window.trading_dates)
    outside = sorted(positive - planned)
    if outside:
        raise ValueError(
            "positive_dates contains dates outside the planned window: "
            + ", ".join(value.date().isoformat() for value in outside)
        )

    now = _utc_now()
    with engine.begin() as connection:
        session_row = _load_session_mapping(connection, labeling_session_id)
        _validate_session_config(session_row, config)
        if session_row["completed"]:
            raise ValueError("Cannot save labels to a completed session.")
        tickers = tuple(json.loads(session_row["tickers_json"]))
        if ticker not in tickers:
            raise ValueError(f"Ticker {ticker!r} is not part of the labeling session.")
        current_ticker_position = int(session_row["current_ticker_position"])
        if not 0 <= current_ticker_position < len(tickers):
            raise ValueError("The session current ticker position is outside its ticker list.")
        current_ticker = tickers[current_ticker_position]
        if ticker != current_ticker:
            raise ValueError(
                f"Cannot save ticker {ticker!r}; the session currently points to "
                f"{current_ticker!r}."
            )
        if window.position != int(session_row["current_window_position"]):
            raise ValueError(
                f"Cannot save window {window.position}; the session currently points to "
                f"window {session_row['current_window_position']}."
            )

        label_values = [
            {
                "provider": session_row["provider"],
                "ticker": ticker,
                "trading_date": trading_date.date(),
                "label_family": session_row["label_family"],
                "label": int(trading_date in positive),
                "labeling_session_id": labeling_session_id,
                "labeled_at": now,
            }
            for trading_date in window.trading_dates
        ]
        _upsert_labels(connection, label_values)

        progress_values: dict[str, Any] = {
            "updated_at": now,
            "atr_stop_multiple": config.atr_stop_multiple,
            "reward_risk_ratio": config.reward_risk_ratio,
        }
        if next_ticker_position is not None:
            if not 0 <= next_ticker_position < len(tickers):
                raise ValueError("next_ticker_position is outside the session ticker list.")
            progress_values["current_ticker_position"] = next_ticker_position
        if next_window_position is not None:
            if next_window_position < 0:
                raise ValueError("next_window_position must not be negative.")
            progress_values["current_window_position"] = next_window_position
        if completed is not None:
            progress_values["completed"] = completed
        connection.execute(
            labeling_sessions.update()
            .where(labeling_sessions.c.labeling_session_id == labeling_session_id)
            .values(**progress_values)
        )

    return load_labeling_session(engine=engine, labeling_session_id=labeling_session_id)


def plan_labeling_windows(
    trading_dates: pd.DatetimeIndex | Sequence[date | str | pd.Timestamp],
    *,
    config: LabelingConfig,
    labeling_end_date: date | str | pd.Timestamp,
    labeling_start_date: date | str | pd.Timestamp | None = None,
) -> tuple[LabelingWindow, ...]:
    """Build deterministic full windows with enough in-bound future outcome data.

    Windows use trading-session positions rather than calendar durations. Each
    start advances by ``config.step_size`` and contains exactly
    ``config.window_size`` sessions. A window is emitted only when its final
    candle and every configured forward horizon are on or before
    ``labeling_end_date``. This lets callers cap labeling at validation without
    reading locked-test outcomes.
    """

    dates = _validated_trading_dates(trading_dates)
    end_date = _to_timestamp(labeling_end_date)
    allowed_dates = dates[dates <= end_date]
    if labeling_start_date is None:
        first_position = 0
    else:
        start_date = _to_timestamp(labeling_start_date)
        if start_date > end_date:
            raise ValueError("labeling_start_date must not be later than labeling_end_date.")
        first_position = int(allowed_dates.searchsorted(start_date, side="left"))
    windows: list[LabelingWindow] = []
    position = 0
    start_position = first_position
    while True:
        end_position = start_position + config.window_size - 1
        future_end_position = end_position + config.forward_horizon
        if future_end_position >= len(allowed_dates):
            break
        selected_dates = tuple(
            pd.Timestamp(value) for value in allowed_dates[start_position : end_position + 1]
        )
        windows.append(
            LabelingWindow(
                position=position,
                start_position=start_position,
                end_position=end_position,
                start_date=selected_dates[0],
                end_date=selected_dates[-1],
                trading_dates=selected_dates,
            )
        )
        start_position += config.step_size
        position += 1
    return tuple(windows)


def prepare_labeling_frame(prices: pd.DataFrame, *, config: LabelingConfig) -> pd.DataFrame:
    """Validate one ordered instrument and add chart indicators for labeling.

    EMA 10/20/50, raw-price ATR, and retrospective pivot flags are calculated
    through the existing indicator APIs. The returned frame is a copy and keeps
    the original ``DatetimeIndex``.
    """

    required_columns = {"open", "high", "low", "close", "volume"}
    missing = sorted(required_columns - set(prices.columns))
    if missing:
        raise ValueError(f"Missing labeling price columns: {', '.join(missing)}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must use a DatetimeIndex for one instrument.")
    if prices.index.has_duplicates:
        raise ValueError("prices index must be unique.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be ordered by trading date.")

    frame = prices.copy()
    for length in EMA_COLORS:
        frame[f"ema_{length}"] = ema(frame["close"], length=length)
    frame["atr"] = atr(frame[["high", "low", "close"]], length=config.atr_length)
    pivots = pivot_points_high_low(
        frame,
        high_left=config.pivot_high_left,
        high_right=config.pivot_high_right,
        low_left=config.pivot_low_left,
        low_right=config.pivot_low_right,
    )
    frame["pivot_high"] = pivots["pivot_high"]
    frame["pivot_low"] = pivots["pivot_low"]
    return frame


def slice_chart_context(
    frame: pd.DataFrame,
    *,
    window: LabelingWindow,
    config: LabelingConfig,
) -> pd.DataFrame:
    """Return one step of pan context on each side plus forward-outcome rows."""

    start = max(0, window.start_position - config.step_size)
    end = min(
        len(frame),
        window.end_position + config.step_size + config.forward_horizon + 1,
    )
    return frame.iloc[start:end].copy()


def calculate_forward_outcomes(
    close: pd.Series,
    atr_values: pd.Series,
    *,
    config: LabelingConfig,
) -> ForwardOutcomeMatrices:
    """Calculate commission-aware outcomes from each close over future horizons.

    Entry commission increases the close-based entry cost and exit commission
    reduces future close proceeds. The same net profit is expressed as a return,
    in ATR units, and in configured stop-distance risk units. Unavailable future
    rows or non-positive ATR denominators remain missing.
    """

    if not close.index.equals(atr_values.index):
        raise ValueError("close and atr_values must have identical indexes.")
    if close.index.has_duplicates:
        raise ValueError("close and atr_values indexes must be unique.")

    buy = close.astype("float64")
    entry_cost = buy * (1.0 + config.commission_rate)
    net_rows: list[pd.Series] = []
    atr_rows: list[pd.Series] = []
    risk_rows: list[pd.Series] = []
    horizons = range(1, config.forward_horizon + 1)
    for horizon in horizons:
        sell = buy.shift(-horizon)
        exit_proceeds = sell * (1.0 - config.commission_rate)
        net_pnl = exit_proceeds - entry_cost
        net_rows.append((exit_proceeds / entry_cost - 1.0).rename(horizon))
        atr_rows.append((net_pnl / atr_values).where(atr_values > 0).rename(horizon))
        risk_distance = atr_values * config.atr_stop_multiple
        risk_rows.append((net_pnl / risk_distance).where(risk_distance > 0).rename(horizon))

    return ForwardOutcomeMatrices(
        net_return=pd.DataFrame(net_rows, columns=close.index),
        atr_units=pd.DataFrame(atr_rows, columns=close.index),
        risk_units=pd.DataFrame(risk_rows, columns=close.index),
    )


def risk_guide_for_date(
    frame: pd.DataFrame,
    *,
    trading_date: date | str | pd.Timestamp,
    config: LabelingConfig,
) -> RiskGuide | None:
    """Return the close-based visual stop and target for one hovered candle."""

    timestamp = _to_timestamp(trading_date)
    if timestamp not in frame.index:
        raise ValueError(f"Trading date {timestamp.date()} is absent from the chart frame.")
    row = frame.loc[timestamp]
    entry = float(row["close"])
    atr_value = float(row["atr"])
    if not math.isfinite(entry) or not math.isfinite(atr_value) or atr_value <= 0:
        return None
    stop_distance = config.atr_stop_multiple * atr_value
    location = int(frame.index.get_loc(timestamp))
    end_location = min(location + config.forward_horizon, len(frame) - 1)
    return RiskGuide(
        trading_date=timestamp,
        end_date=pd.Timestamp(frame.index[end_location]),
        entry=entry,
        atr=atr_value,
        stop=entry - stop_distance,
        take_profit=entry + config.reward_risk_ratio * stop_distance,
    )


def _nan_to_none(values: Any) -> np.ndarray:
    """Return an object array with ``NaN`` replaced by ``None``.

    Plotly renders both ``NaN`` and ``None`` as gaps, but ``None`` serializes to
    JSON ``null`` while ``NaN`` is not JSON compliant and triggers a
    ``jupyter_client`` serialization warning when pushed to a ``FigureWidget``.
    """

    array = np.asarray(values, dtype=float)
    result = array.astype(object)
    result[np.isnan(array)] = None
    return result


def build_labeling_figure(
    frame: pd.DataFrame,
    *,
    window: LabelingWindow,
    selected_dates: Iterable[date | str | pd.Timestamp],
    config: LabelingConfig,
    heatmap_mode: HeatmapMode | None = None,
) -> Figure:
    """Build the labeling chart before notebook callbacks are attached.

    The figure contains candlesticks and indicators, directional volume, and one
    selected forward-outcome heatmap. Empty named stop and take-profit traces are
    included so a ``FigureWidget`` hover callback can update them in place.
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mode = heatmap_mode or config.default_heatmap_mode
    if mode not in SUPPORTED_HEATMAP_MODES:
        raise ValueError(f"Unsupported heatmap mode: {mode!r}.")
    planned_dates = set(window.trading_dates)
    missing_planned = sorted(planned_dates - set(frame.index))
    if missing_planned:
        raise ValueError("The chart frame does not contain every planned-window candle.")
    selected = {_to_timestamp(value) for value in selected_dates}
    outside = sorted(selected - planned_dates)
    if outside:
        raise ValueError("selected_dates contains dates outside the planned window.")

    outcomes = calculate_forward_outcomes(frame["close"], frame["atr"], config=config)
    heatmap = outcomes.for_mode(mode)
    z_limit = _robust_symmetric_limit(heatmap)
    volume_colors = np.where(frame["close"] >= frame["open"], "#26a69a", "#ef5350")

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.68, 0.12, 0.20],
    )
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            line_width=1,
            name="OHLC",
            hoverinfo="none",
        ),
        row=1,
        col=1,
    )
    for length, color in EMA_COLORS.items():
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=_nan_to_none(frame[f"ema_{length}"].to_numpy()),
                mode="lines",
                line={"color": color, "width": 1.4},
                name=f"EMA {length}",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    marker_x, marker_y = _selected_marker_coordinates(frame, selected)
    figure.add_trace(
        go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            marker={
                "symbol": "triangle-up",
                "size": 12,
                "color": "#39da89",
                "line": {"color": "#111111", "width": 1},
            },
            name=SELECTED_TRACE_NAME,
            customdata=[value.isoformat() for value in marker_x],
            hovertemplate="Selected %{x|%Y-%m-%d}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    for name, color, dash in (
        (STOP_TRACE_NAME, "#eb4343", "dash"),
        (TAKE_PROFIT_TRACE_NAME, "#39da89", "dash"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line={"color": color, "width": 2, "dash": dash},
                name=name,
                visible=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=frame.index,
            y=frame["volume"],
            marker_color=volume_colors,
            name="Volume",
            hovertemplate="%{x|%Y-%m-%d}<br>Volume %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=heatmap.columns,
            y=heatmap.index,
            z=_nan_to_none(heatmap.to_numpy()),
            colorscale="RdYlGn",
            zmid=0,
            zmin=-z_limit,
            zmax=z_limit,
            colorbar={"title": _heatmap_colorbar_title(mode), "len": 0.23, "y": 0.10},
            name="Forward outcomes",
            hovertemplate=_heatmap_hover_template(mode),
        ),
        row=3,
        col=1,
    )

    for trading_date, value in frame.loc[frame["pivot_low"].fillna(False), "low"].items():
        add_pivot_annotation(
            figure,
            x=pd.Timestamp(trading_date),
            y=float(value),
            kind="low",
        )
    for trading_date, value in frame.loc[frame["pivot_high"].fillna(False), "high"].items():
        add_pivot_annotation(
            figure,
            x=pd.Timestamp(trading_date),
            y=float(value),
            kind="high",
        )

    figure.update_layout(
        height=650,
        width=1400,
        hovermode="x unified",
        hoverlabel={"bgcolor": "rgba(250, 250, 250, 0.45)"},
        clickmode="event",
        dragmode="pan",
        margin={"l": 60, "r": 40, "t": 45, "b": 35},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        uirevision="labeling-workflow",
        paper_bgcolor="white",
        plot_bgcolor="white",
        modebar={"remove": ["select2d", "lasso2d"]},
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor="#e6e6e6",
        zeroline=False,
        showline=True,
        linecolor="#c7c7c7",
        linewidth=1,
        mirror=True,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="#e6e6e6",
        zeroline=False,
        showline=True,
        linecolor="#c7c7c7",
        linewidth=1,
        mirror=True,
    )
    figure.update_xaxes(
        range=[window.start_date, window.end_date],
        rangeslider_visible=False,
        rangebreaks=[{"bounds": ["sat", "mon"]}],
        row=1,
        col=1,
    )
    figure.update_yaxes(title_text="Price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    figure.update_yaxes(title_text="Sessions", autorange="reversed", dtick=1, row=3, col=1)

    figure.add_annotation(
        name=PRICE_HUD_NAME,
        xref="x domain",
        yref="y domain",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        text="",
        showarrow=False,
        align="left",
        font={"family": "Courier New, monospace", "size": 16, "color": "#333333"},
        bgcolor="white",
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=4,
        row=1,
        col=1,
    )
    return figure


def add_pivot_annotation(
    figure: Figure,
    *,
    x: pd.Timestamp,
    y: float,
    kind: Literal["low", "high"],
) -> None:
    """Add a bounded-offset pivot annotation preserving the established style."""

    if kind not in ("low", "high"):
        raise ValueError(f"kind must be 'low' or 'high'; got {kind!r}.")
    font_color = "#39da89" if kind == "low" else "#eb4343"
    raw_offset = 5 * np.log2(max(y, np.finfo(float).tiny))
    offset = float(np.clip(raw_offset, 12, 45))
    figure.add_annotation(
        x=x,
        y=y,
        xref="x",
        yref="y",
        text=f"{y:.1f}",
        showarrow=True,
        font={
            "family": "Courier New, monospace",
            "size": 10,
            "color": font_color,
            "weight": 900,
        },
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=offset if kind == "low" else -offset,
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=3,
        bgcolor="#484848",
        opacity=0.8,
        row=1,
        col=1,
    )


def update_selected_trace(
    figure: Figure,
    frame: pd.DataFrame,
    selected_dates: Iterable[date | str | pd.Timestamp],
) -> None:
    """Update the dedicated selected-entry trace in an existing figure."""

    selected = {_to_timestamp(value) for value in selected_dates}
    marker_x, marker_y = _selected_marker_coordinates(frame, selected)
    trace = _trace_by_name(figure, SELECTED_TRACE_NAME)
    trace.x = marker_x
    trace.y = marker_y
    trace.customdata = [value.isoformat() for value in marker_x]


def update_price_hud(
    figure: Figure,
    frame: pd.DataFrame,
    *,
    trading_date: date | str | pd.Timestamp | None,
) -> None:
    """Update the pinned upper-left OHLC/date annotation in an existing figure."""

    annotation = next(
        (item for item in figure.layout.annotations if item.name == PRICE_HUD_NAME),
        None,
    )
    if annotation is None:
        return
    if trading_date is None:
        annotation.text = ""
        return
    timestamp = _to_timestamp(trading_date)
    if timestamp not in frame.index:
        annotation.text = ""
        return
    row = frame.loc[timestamp]
    color = "#26a69a" if row["close"] > row["open"] else "#ef5350"

    def value(number: float) -> str:
        return f'<span style="color:{color}">{number:.2f}</span>'

    change_percent = round(100 * (row["close"] / row["open"] - 1), 2)
    annotation.text = (
        f"<b>{timestamp.date()} "
        f"O {value(row['open'])}  H {value(row['high'])}  "
        f"L {value(row['low'])}  C {value(row['close'])}  "
        f'<span style="color:{color}">({change_percent:.2f} %)</span></b>'
    )


def update_risk_guide_traces(figure: Figure, guide: RiskGuide | None) -> None:
    """Show or hide the hover-driven stop and take-profit traces."""

    stop_trace = _trace_by_name(figure, STOP_TRACE_NAME)
    target_trace = _trace_by_name(figure, TAKE_PROFIT_TRACE_NAME)
    if guide is None:
        stop_trace.visible = False
        target_trace.visible = False
        return
    x_values = [guide.trading_date, guide.end_date]
    stop_trace.x = x_values
    stop_trace.y = [guide.stop, guide.stop]
    stop_trace.visible = True
    target_trace.x = x_values
    target_trace.y = [guide.take_profit, guide.take_profit]
    target_trace.visible = True


def _upsert_labels(connection: Connection, values: list[dict[str, Any]]) -> None:
    for start in range(0, len(values), 100):
        chunk = values[start : start + 100]
        if connection.dialect.name == "sqlite":
            statement = _sqlite_label_upsert(chunk)
        elif connection.dialect.name == "postgresql":
            statement = _postgresql_label_upsert(chunk)
        else:
            raise ValueError(f"Unsupported labeling database dialect: {connection.dialect.name!r}.")
        connection.execute(statement)


def _sqlite_label_upsert(values: list[dict[str, Any]]) -> Any:
    from sqlalchemy.dialects.sqlite import insert

    statement = insert(candle_labels).values(values)
    return statement.on_conflict_do_update(
        index_elements=[
            candle_labels.c.provider,
            candle_labels.c.ticker,
            candle_labels.c.trading_date,
        ],
        set_=_label_update_values(statement),
    )


def _postgresql_label_upsert(values: list[dict[str, Any]]) -> Any:
    from sqlalchemy.dialects.postgresql import insert

    statement = insert(candle_labels).values(values)
    return statement.on_conflict_do_update(
        index_elements=[
            candle_labels.c.provider,
            candle_labels.c.ticker,
            candle_labels.c.trading_date,
        ],
        set_=_label_update_values(statement),
    )


def _label_update_values(statement: Any) -> dict[str, Any]:
    return {
        "label_family": statement.excluded.label_family,
        "label": statement.excluded.label,
        "labeling_session_id": statement.excluded.labeling_session_id,
        "labeled_at": statement.excluded.labeled_at,
    }


def _load_session_mapping(connection: Connection, labeling_session_id: str) -> Any:
    row = (
        connection.execute(
            select(labeling_sessions).where(
                labeling_sessions.c.labeling_session_id == labeling_session_id
            )
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise ValueError(f"Unknown labeling session: {labeling_session_id!r}.")
    return row


def _session_from_mapping(row: Any) -> LabelingSession:
    config = LabelingConfig(
        window_size=int(row["window_size"]),
        step_size=int(row["step_size"]),
        forward_horizon=int(row["forward_horizon"]),
        atr_length=int(row["atr_length"]),
        atr_stop_multiple=float(row["atr_stop_multiple"]),
        reward_risk_ratio=float(row["reward_risk_ratio"]),
        commission_rate=float(row["commission_rate"]),
        pivot_high_left=int(row["pivot_high_left"]),
        pivot_high_right=int(row["pivot_high_right"]),
        pivot_low_left=int(row["pivot_low_left"]),
        pivot_low_right=int(row["pivot_low_right"]),
        default_heatmap_mode=row["default_heatmap_mode"],
    )
    return LabelingSession(
        labeling_session_id=row["labeling_session_id"],
        provider=row["provider"],
        tickers=tuple(json.loads(row["tickers_json"])),
        label_family=row["label_family"],
        labeling_start_date=row["labeling_start_date"],
        labeling_end_date=row["labeling_end_date"],
        current_ticker_position=int(row["current_ticker_position"]),
        current_window_position=int(row["current_window_position"]),
        config=config,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed=bool(row["completed"]),
    )


def _validate_session_config(row: Any, config: LabelingConfig) -> None:
    fixed_fields = (
        "window_size",
        "step_size",
        "forward_horizon",
        "atr_length",
        "commission_rate",
        "pivot_high_left",
        "pivot_high_right",
        "pivot_low_left",
        "pivot_low_right",
        "default_heatmap_mode",
    )
    mismatches = [name for name in fixed_fields if getattr(config, name) != row[name]]
    if mismatches:
        raise ValueError(
            "Only atr_stop_multiple and reward_risk_ratio may change within a session; "
            f"fixed fields changed: {', '.join(mismatches)}."
        )


def _config_values(config: LabelingConfig) -> dict[str, Any]:
    return {
        "window_size": config.window_size,
        "step_size": config.step_size,
        "forward_horizon": config.forward_horizon,
        "atr_length": config.atr_length,
        "atr_stop_multiple": config.atr_stop_multiple,
        "reward_risk_ratio": config.reward_risk_ratio,
        "commission_rate": config.commission_rate,
        "pivot_high_left": config.pivot_high_left,
        "pivot_high_right": config.pivot_high_right,
        "pivot_low_left": config.pivot_low_left,
        "pivot_low_right": config.pivot_low_right,
        "default_heatmap_mode": config.default_heatmap_mode,
    }


def _validated_trading_dates(
    values: pd.DatetimeIndex | Sequence[date | str | pd.Timestamp],
) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values))
    if dates.has_duplicates:
        raise ValueError("trading_dates must be unique.")
    if not dates.is_monotonic_increasing:
        raise ValueError("trading_dates must be ordered.")
    return dates


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_required_text(ticker, name="ticker") for ticker in tickers)
    if not normalized:
        raise ValueError("tickers must contain at least one symbol.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("tickers must not contain duplicates.")
    return normalized


def _required_text(value: str, *, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _to_date(value: date | str | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return _to_timestamp(value).date()


def _to_timestamp(value: date | str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _selected_marker_coordinates(
    frame: pd.DataFrame,
    selected: set[pd.Timestamp],
) -> tuple[list[pd.Timestamp], list[float]]:
    available = sorted(selected & set(frame.index))
    marker_y: list[float] = []
    for timestamp in available:
        row = frame.loc[timestamp]
        atr_value = float(row["atr"])
        low = float(row["low"])
        offset = 0.25 * atr_value if math.isfinite(atr_value) and atr_value > 0 else 0.01 * low
        marker_y.append(low - offset)
    return available, marker_y


def _robust_symmetric_limit(values: pd.DataFrame) -> float:
    finite = np.abs(values.to_numpy(dtype="float64"))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.quantile(finite, 0.98))
    return limit if limit > 0 else 1.0


def _heatmap_colorbar_title(mode: HeatmapMode) -> str:
    return {
        "net_return": "Net return",
        "atr_units": "ATR units",
        "risk_units": "Risk units",
    }[mode]


def _heatmap_hover_template(mode: HeatmapMode) -> str:
    value_format = ".2%" if mode == "net_return" else ".2f"
    return (
        "Entry %{x|%Y-%m-%d}<br>Horizon %{y} sessions"
        f"<br>{_heatmap_colorbar_title(mode)} %{{z:{value_format}}}<extra></extra>"
    )


def _trace_by_name(figure: Figure, name: str) -> Any:
    matches = [trace for trace in figure.data if trace.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one trace named {name!r}; found {len(matches)}.")
    return matches[0]
