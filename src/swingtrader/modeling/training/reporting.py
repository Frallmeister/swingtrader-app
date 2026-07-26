"""Artifact serialization and dependency-free SVG reports for model evaluation."""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from swingtrader.modeling.training.evaluation import EvaluationReport


def write_evaluation_artifacts(
    report: EvaluationReport,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Write deterministic JSON, CSV, Markdown, and SVG artifacts for one report."""
    if not isinstance(report, EvaluationReport):
        raise TypeError("Artifact writing requires an EvaluationReport.")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    plots = root / "plots"
    plots.mkdir(exist_ok=True)

    paths: list[Path] = []
    manifest_path = root / "summary.json"
    manifest_path.write_text(
        json.dumps(report.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.append(manifest_path)

    tables = {
        "predictions.csv.gz": report.predictions.reset_index(),
        "per_date_metrics.csv": report.per_date_metrics,
        "calibration.csv": report.calibration,
        "score_quantiles.csv": report.score_quantiles,
        "score_quantiles_by_date.csv": report.score_quantiles_by_date,
        "top_k_by_date.csv": report.top_k_by_date,
        "random_top_k_by_date.csv": report.random_top_k_by_date,
        "feature_missingness.csv": report.feature_missingness,
    }
    for name, table in tables.items():
        path = root / name
        compression: str | Mapping[str, object] | None = None
        if path.suffix == ".gz":
            compression = {"method": "gzip", "mtime": 0}
        table.to_csv(path, index=False, compression=compression)
        paths.append(path)

    report_path = root / "report.md"
    report_path.write_text(_markdown_report(report), encoding="utf-8")
    paths.append(report_path)

    plot_specs = (
        (
            plots / "calibration.svg",
            report.calibration["mean_score"],
            report.calibration["observed_rate"],
            "Mean predicted score",
            "Observed positive rate",
            "Calibration",
            (0.0, 1.0),
            (0.0, 1.0),
            True,
        ),
        (
            plots / "score_quantile_positive_rate.svg",
            report.score_quantiles["score_quantile"],
            report.score_quantiles["positive_rate"],
            "Daily score quantile",
            "Mean daily positive rate",
            "Positive rate by score quantile",
            None,
            (0.0, 1.0),
            False,
        ),
        (
            plots / "score_quantile_return.svg",
            report.score_quantiles["score_quantile"],
            report.score_quantiles["mean_ranking_return"],
            "Daily score quantile",
            "Mean daily ranking return",
            "Return by score quantile",
            None,
            None,
            False,
        ),
    )
    for path, x, y, x_label, y_label, title, x_range, y_range, reference in plot_specs:
        _write_line_svg(
            path,
            x=x,
            y=y,
            x_label=x_label,
            y_label=y_label,
            title=title,
            x_range=x_range,
            y_range=y_range,
            reference_diagonal=reference,
        )
        paths.append(path)

    distribution_path = plots / "top_k_return_distribution.svg"
    model_returns, random_returns = _paired_top_k_returns(report)
    _write_distribution_svg(
        distribution_path,
        model=model_returns,
        random=random_returns,
    )
    paths.append(distribution_path)
    return tuple(paths)


def _paired_top_k_returns(report: EvaluationReport) -> tuple[pd.Series, pd.Series]:
    paired = report.per_date_metrics.dropna(
        subset=["top_k_mean_return", "random_top_k_mean_return"]
    )
    return paired["top_k_mean_return"], paired["random_top_k_mean_return"]


def _markdown_report(report: EvaluationReport) -> str:
    lines = [
        f"# {report.split.title()} Evaluation Report",
        "",
        "## Dataset Context",
        "",
        _mapping_table(report.dataset_context),
        "",
        "## Evaluation Configuration",
        "",
        _mapping_table(report.config.to_manifest()),
        "",
        "## Ranking Outcome",
        "",
        (
            f"Source target column: `{report.ranking_return_column}`."
            if report.ranking_return_column is not None
            else "No continuous ranking-return target was supplied."
        ),
        "",
        "## Interpretation Boundary",
        "",
        (
            "`ranking_return` is a research diagnostic outcome. It does not apply "
            "next-session entry assumptions and excludes transaction costs, spreads, "
            "slippage, stop-loss or take-profit execution, position sizing, and "
            "portfolio constraints. Do not interpret it as executable strategy P&L."
        ),
        "",
        "## Aggregate Metrics",
        "",
        _mapping_table(report.aggregate_metrics),
        "",
        "## Artifact Tables",
        "",
        "- `predictions.csv.gz`: canonical row-level prediction frame.",
        "- `per_date_metrics.csv`: daily classification and ranking results.",
        "- `calibration.csv`: fixed-width probability buckets.",
        "- `score_quantiles.csv`: equal-weight summaries across dates.",
        "- `score_quantiles_by_date.csv`: daily cross-sectional score quantiles.",
        "- `top_k_by_date.csv`: model-selected top-k results by date.",
        "- `random_top_k_by_date.csv`: date-matched random comparison.",
        "- `feature_missingness.csv`: evaluation-split missingness context.",
        "",
        "## Plots",
        "",
        "- [Calibration](plots/calibration.svg)",
        "- [Positive rate by score quantile](plots/score_quantile_positive_rate.svg)",
        "- [Return by score quantile](plots/score_quantile_return.svg)",
        "- [Daily top-k return distributions](plots/top_k_return_distribution.svg)",
        "",
    ]
    return "\n".join(lines)


def _mapping_table(values: Mapping[str, object]) -> str:
    lines = ["| Name | Value |", "| --- | ---: |"]
    for name, value in values.items():
        if isinstance(value, float):
            rendered = f"{value:.8g}" if math.isfinite(value) else "undefined"
        else:
            rendered = str(value)
        lines.append(f"| `{name}` | {rendered} |")
    return "\n".join(lines)


def _write_line_svg(
    path: Path,
    *,
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    reference_diagonal: bool,
) -> None:
    x_values = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
    y_values = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    width, height = 760, 440
    left, right, top, bottom = 82, 28, 54, 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min, x_max = _axis_range(x_values, fixed=x_range)
    y_min, y_max = _axis_range(y_values, fixed=y_range)

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = _axes(
        width=width,
        height=height,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )
    if reference_diagonal:
        low = max(x_min, y_min)
        high = min(x_max, y_max)
        if low < high:
            elements.append(_svg_line(px(low), py(low), px(high), py(high), dash="6 5"))
    if len(x_values):
        points = " ".join(
            f"{px(x_value):.2f},{py(y_value):.2f}"
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="currentColor" stroke-width="2"/>'
        )
        elements.extend(
            f'<circle cx="{px(x_value):.2f}" cy="{py(y_value):.2f}" r="3" fill="currentColor"/>'
            for x_value, y_value in zip(x_values, y_values, strict=True)
        )
    else:
        elements.append('<text x="380" y="220" text-anchor="middle">No finite values</text>')
    _write_svg(path, width=width, height=height, elements=elements)


def _write_distribution_svg(
    path: Path,
    *,
    model: pd.Series,
    random: pd.Series,
) -> None:
    model_values = _finite_sorted(model)
    random_values = _finite_sorted(random)
    all_values = np.concatenate([model_values, random_values])
    width, height = 760, 440
    left, right, top, bottom = 82, 28, 54, 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = _axis_range(all_values, fixed=None)
    y_min, y_max = 0.0, 1.0

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def py(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = _axes(
        width=width,
        height=height,
        left=left,
        top=top,
        plot_width=plot_width,
        plot_height=plot_height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label="Mean selected ranking return",
        y_label="Empirical cumulative fraction",
        title="Daily top-k return distributions",
    )
    for values, dash in ((model_values, None), (random_values, "6 5")):
        if not len(values):
            continue
        points = " ".join(
            f"{px(value):.2f},{py(position / len(values)):.2f}"
            for position, value in enumerate(values, start=1)
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        elements.append(
            f'<polyline points="{points}" fill="none" stroke="currentColor" '
            f'stroke-width="2"{dash_attribute}/>'
        )
    if not len(all_values):
        elements.append('<text x="380" y="220" text-anchor="middle">No ranking returns</text>')
    elements.extend(
        [
            _svg_line(width - 218, 75, width - 174, 75),
            f'<text x="{width - 166}" y="80">model</text>',
            _svg_line(width - 218, 97, width - 174, 97, dash="6 5"),
            f'<text x="{width - 166}" y="102">random</text>',
        ]
    )
    _write_svg(path, width=width, height=height, elements=elements)


def _axes(
    *,
    width: int,
    height: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    x_label: str,
    y_label: str,
    title: str,
) -> list[str]:
    elements = [
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" '
        f'font-size="18">{html.escape(title)}</text>',
    ]
    for value in np.linspace(x_min, x_max, 5):
        x_position = left + (value - x_min) / (x_max - x_min) * plot_width
        elements.append(_svg_line(x_position, top, x_position, top + plot_height, css_class="grid"))
        elements.append(
            f'<text x="{x_position:.2f}" y="{top + plot_height + 23}" '
            f'text-anchor="middle">{value:.3g}</text>'
        )
    for value in np.linspace(y_min, y_max, 5):
        y_position = top + (y_max - value) / (y_max - y_min) * plot_height
        elements.append(
            _svg_line(left, y_position, left + plot_width, y_position, css_class="grid")
        )
        elements.append(
            f'<text x="{left - 10}" y="{y_position + 5:.2f}" text-anchor="end">{value:.3g}</text>'
        )
    elements.extend(
        [
            _svg_line(left, top + plot_height, left + plot_width, top + plot_height),
            _svg_line(left, top, left, top + plot_height),
            f'<text x="{width / 2:.0f}" y="{height - 18}" text-anchor="middle">'
            f"{html.escape(x_label)}</text>",
            f'<text x="20" y="{height / 2:.0f}" text-anchor="middle" '
            f'transform="rotate(-90 20 {height / 2:.0f})">{html.escape(y_label)}</text>',
        ]
    )
    return elements


def _axis_range(
    values: np.ndarray,
    *,
    fixed: tuple[float, float] | None,
) -> tuple[float, float]:
    if fixed is not None:
        minimum, maximum = fixed
    elif len(values):
        minimum, maximum = float(values.min()), float(values.max())
    else:
        minimum, maximum = 0.0, 1.0
    if minimum == maximum:
        padding = max(abs(minimum) * 0.1, 0.1)
        minimum -= padding
        maximum += padding
    else:
        padding = (maximum - minimum) * 0.05
        if fixed is None:
            minimum -= padding
            maximum += padding
    return minimum, maximum


def _finite_sorted(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    return np.sort(numeric[np.isfinite(numeric)])


def _svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    dash: str | None = None,
    css_class: str | None = None,
) -> str:
    attributes = []
    if dash:
        attributes.append(f'stroke-dasharray="{dash}"')
    if css_class:
        attributes.append(f'class="{css_class}"')
    suffix = " " + " ".join(attributes) if attributes else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"{suffix}/>'


def _write_svg(
    path: Path,
    *,
    width: int,
    height: int,
    elements: list[str],
) -> None:
    content = "\n".join(elements)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<style>"
        "text{font-family:system-ui,sans-serif;font-size:13px;fill:#1f2937}"
        "line,polyline{stroke:#1f2937}.grid{stroke:#d1d5db;stroke-width:1}"
        "</style>\n"
        f"{content}\n"
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
