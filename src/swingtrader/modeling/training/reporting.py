"""Artifact serialization and report generation for model evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from swingtrader.modeling.training.evaluation import EvaluationReport
from swingtrader.modeling.training._plotting import (
    write_distribution_plot,
    write_line_plot,
)


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
            False,
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
            True,
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
            True,
        ),
    )
    for (
        path,
        x,
        y,
        x_label,
        y_label,
        title,
        x_range,
        y_range,
        reference,
        integer_x,
    ) in plot_specs:
        write_line_plot(
            path,
            x=x,
            y=y,
            x_label=x_label,
            y_label=y_label,
            title=title,
            x_range=x_range,
            y_range=y_range,
            reference_diagonal=reference,
            integer_x=integer_x,
        )
        paths.append(path)
    distribution_path = plots / "top_k_return_distribution.svg"
    model_returns, random_returns = _paired_top_k_returns(report)
    write_distribution_plot(
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
