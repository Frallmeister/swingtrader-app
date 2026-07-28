"""Internal matplotlib-backed SVG plots for model-evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

_FIGURE_SIZE = (7.6, 4.4)
_RC_PARAMS: dict[Any, Any] = {
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "svg.hashsalt": "swingtrader-model-evaluation",
}


def write_line_plot(
    path: Path,
    *,
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    reference_diagonal: bool = False,
    integer_x: bool = False,
) -> None:
    """Write one deterministic SVG line plot from finite paired values."""
    x_values, y_values = _finite_pairs(x, y)
    with matplotlib.rc_context(_RC_PARAMS):
        figure, axis = _new_figure()
        try:
            if len(x_values):
                axis.plot(x_values, y_values, marker="o")
            else:
                _empty_message(axis, "No finite values")
            _configure_axis(
                axis,
                title=title,
                x_label=x_label,
                y_label=y_label,
                x_range=x_range,
                y_range=y_range,
                has_values=bool(len(x_values)),
            )
            if integer_x:
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            if reference_diagonal:
                _add_reference_diagonal(axis)
            _save_figure(figure, path)
        finally:
            figure.clear()


def write_distribution_plot(
    path: Path,
    *,
    model: pd.Series,
    random: pd.Series,
) -> None:
    """Write deterministic empirical CDFs for model and random top-k returns."""
    model_values = _finite_sorted(model)
    random_values = _finite_sorted(random)
    with matplotlib.rc_context(_RC_PARAMS):
        figure, axis = _new_figure()
        try:
            _plot_ecdf(axis, model_values, label="model")
            _plot_ecdf(axis, random_values, label="random", linestyle="--")
            has_values = bool(len(model_values) or len(random_values))
            if not has_values:
                _empty_message(axis, "No ranking returns")
            _configure_axis(
                axis,
                title="Daily top-k return distributions",
                x_label="Mean selected ranking return",
                y_label="Empirical cumulative fraction",
                x_range=None,
                y_range=(0.0, 1.0),
                has_values=has_values,
            )
            if has_values:
                axis.legend()
            _save_figure(figure, path)
        finally:
            figure.clear()


def _new_figure() -> tuple[Figure, Axes]:
    figure = Figure(figsize=_FIGURE_SIZE)
    FigureCanvasSVG(figure)
    return figure, figure.subplots()


def _finite_pairs(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid], y_values[valid]


def _finite_sorted(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    return np.sort(numeric[np.isfinite(numeric)])


def _plot_ecdf(
    axis: Axes,
    values: np.ndarray,
    *,
    label: str,
    linestyle: str = "-",
) -> None:
    if not len(values):
        return
    cumulative_fraction = np.arange(1, len(values) + 1, dtype="float64") / len(values)
    axis.plot(
        values,
        cumulative_fraction,
        drawstyle="steps-post",
        label=label,
        linestyle=linestyle,
    )


def _configure_axis(
    axis: Axes,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    has_values: bool,
) -> None:
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    if x_range is not None:
        axis.set_xlim(*x_range)
    elif not has_values:
        axis.set_xlim(0.0, 1.0)
    if y_range is not None:
        axis.set_ylim(*y_range)
    elif not has_values:
        axis.set_ylim(0.0, 1.0)


def _add_reference_diagonal(axis: Axes) -> None:
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    low = max(x_min, y_min)
    high = min(x_max, y_max)
    if low < high:
        axis.plot([low, high], [low, high], linestyle="--", zorder=1)


def _empty_message(axis: Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, format="svg", metadata={"Date": None})
