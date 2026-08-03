"""Thin MLflow adapter for executing repository-defined experiments.

The domain contracts in :mod:`swingtrader.modeling.experiments.contracts` do
not import MLflow. This module loads the optional dependency only when a run is
started, translates an :class:`~swingtrader.modeling.experiments.ExperimentSpec`
into tracked parameters and artifacts, and exposes a small run handle for
metrics and generated files.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from numbers import Real
from pathlib import Path
from types import ModuleType

from swingtrader.modeling.experiments.contracts import ExperimentSpec


@dataclass(frozen=True, slots=True)
class DatasetSplitSummary:
    """Summarize one materialized dataset split without storing its rows."""

    rows: int
    ticker_count: int
    start_date: date | None
    end_date: date | None
    class_prevalence: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.rows, bool) or not isinstance(self.rows, int):
            raise TypeError("Dataset rows must be an integer.")
        if isinstance(self.ticker_count, bool) or not isinstance(self.ticker_count, int):
            raise TypeError("Dataset ticker count must be an integer.")
        if self.rows < 0:
            raise ValueError("Dataset rows must not be negative.")
        if self.ticker_count < 0:
            raise ValueError("Dataset ticker count must not be negative.")
        if self.ticker_count > self.rows:
            raise ValueError("Dataset ticker count must not exceed its row count.")
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("Dataset summary dates must be provided together.")
        for field_name, value in (
            ("Dataset start date", self.start_date),
            ("Dataset end date", self.end_date),
        ):
            if value is not None and type(value) is not date:
                raise TypeError(f"{field_name} must be a datetime.date.")
        if self.rows == 0 and (
            self.ticker_count != 0
            or self.start_date is not None
            or self.class_prevalence is not None
        ):
            raise ValueError(
                "An empty dataset summary must not declare tickers, dates, or prevalence."
            )
        if self.rows > 0 and self.ticker_count == 0:
            raise ValueError("A non-empty dataset summary must declare at least one ticker.")
        if self.rows > 0 and self.start_date is None:
            raise ValueError("A non-empty dataset summary must declare its date range.")
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("Dataset summary start date must not follow its end date.")
        if self.class_prevalence is not None:
            if isinstance(self.class_prevalence, bool) or not isinstance(
                self.class_prevalence, Real
            ):
                raise TypeError("Class prevalence must be a real number.")
            if not math.isfinite(self.class_prevalence):
                raise ValueError("Class prevalence must be finite.")
            if not 0.0 <= self.class_prevalence <= 1.0:
                raise ValueError("Class prevalence must be between zero and one.")


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Hold train, validation, and test summaries for one experiment run."""

    train: DatasetSplitSummary
    validation: DatasetSplitSummary
    test: DatasetSplitSummary

    def __post_init__(self) -> None:
        for split_name in ("train", "validation", "test"):
            if not isinstance(getattr(self, split_name), DatasetSplitSummary):
                raise TypeError(f"Dataset {split_name} summary must be a DatasetSplitSummary.")


class ExperimentRun:
    """Expose the small subset of run logging needed by training code."""

    __slots__ = ("_mlflow", "run_id")

    def __init__(self, mlflow: ModuleType, run_id: str) -> None:
        self._mlflow = mlflow
        self.run_id = run_id

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log finite numeric metrics for the active run."""
        if step is not None:
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError("Metric step must be an integer.")
            if step < 0:
                raise ValueError("Metric step must not be negative.")

        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Metric names must be non-empty strings.")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Metric {name!r} must be a real number.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Metric {name!r} must be finite.")
            normalized[name] = numeric_value
        self._mlflow.log_metrics(normalized, step=step)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        """Log one generated report, plot, or other file."""
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        self._mlflow.log_artifact(str(artifact), artifact_path=artifact_path)


def local_tracking_uri(path: str | Path = "mlflow.db") -> str:
    """Return an absolute SQLite URI for a local MLflow tracking database."""
    database_path = Path(path).expanduser().resolve().as_posix()
    return f"sqlite:///{database_path}"


def resolve_git_revision(repository_root: str | Path | None = None) -> str | None:
    """Return the current Git commit, or ``None`` outside a Git worktree."""
    command = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


@contextmanager
def start_experiment_run(
    spec: ExperimentSpec,
    *,
    experiment_name: str = "swingtrader",
    run_name: str | None = None,
    tracking_uri: str | None = None,
    dataset_summary: DatasetSummary | None = None,
    repository_root: str | Path | None = None,
    tags: Mapping[str, str] | None = None,
) -> Generator[ExperimentRun, None, None]:
    """Start and initialize an MLflow run from an experiment specification.

    The complete canonical manifest is logged under ``manifests/``. Dataset
    summaries record counts, date ranges, and optional class prevalence while
    deliberately omitting complete materialized datasets.
    """
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("MLflow experiment name must be a non-empty string.")
    if run_name is not None and (not isinstance(run_name, str) or not run_name.strip()):
        raise ValueError("MLflow run name must be a non-empty string when provided.")
    if dataset_summary is not None:
        _validate_dataset_summary(spec, dataset_summary)

    mlflow = _import_mlflow()
    resolved_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or local_tracking_uri()
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    git_revision = resolve_git_revision(repository_root)
    resolved_run_name = run_name or f"{spec.name}-{spec.version}-{spec.digest[:8]}"

    with mlflow.start_run(run_name=resolved_run_name) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(_experiment_parameters(spec, git_revision=git_revision))

        run_tags = dict(tags or {})
        run_tags.update(
            {
                "experiment.identifier": spec.identifier,
                "experiment.digest": spec.digest,
            }
        )
        if git_revision is not None:
            run_tags["mlflow.source.git.commit"] = git_revision
        mlflow.set_tags(run_tags)

        if dataset_summary is not None:
            mlflow.log_params(_dataset_parameters(dataset_summary))
            prevalence_metrics = _dataset_metrics(dataset_summary)
            if prevalence_metrics:
                mlflow.log_metrics(prevalence_metrics)

        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory, "experiment.json")
            manifest_path.write_text(f"{spec.to_json()}\n", encoding="utf-8")
            mlflow.log_artifact(str(manifest_path), artifact_path="manifests")

        yield ExperimentRun(mlflow, run_id)


def _import_mlflow() -> ModuleType:
    try:
        return importlib.import_module("mlflow")
    except ModuleNotFoundError as exc:
        if exc.name != "mlflow":
            raise
        raise ModuleNotFoundError(
            "MLflow is required to track experiment runs. Install the modeling extra with "
            "`uv sync --extra modeling`."
        ) from exc


def _parameter_value(value: object) -> bool | int | float | str:
    if value is None:
        return "null"
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _experiment_parameters(
    spec: ExperimentSpec,
    *,
    git_revision: str | None,
) -> dict[str, bool | int | float | str]:
    parameters: dict[str, bool | int | float | str] = {
        "experiment.name": spec.name,
        "experiment.version": spec.version,
        "experiment.identifier": spec.identifier,
        "experiment.digest": spec.digest,
        "feature_set.name": spec.feature_set.name,
        "feature_set.version": spec.feature_set.version,
        "feature_set.identifier": spec.feature_set.identifier,
        "feature_set.digest": spec.feature_set.digest,
        "target_set.name": spec.target_set.name,
        "target_set.version": spec.target_set.version,
        "target_set.identifier": spec.target_set.identifier,
        "target_set.digest": spec.target_set.digest,
        "task.name": spec.task.name,
        "task.type": spec.task.task_type,
        "task.target_column": spec.task.target_column,
        "universe.name": spec.universe.name,
        "universe.version": spec.universe.version,
        "universe.identifier": spec.universe.identifier,
        "universe.digest": spec.universe.digest,
        "universe.provider": spec.universe.provider,
        "universe.ticker_count": len(spec.universe.tickers),
        "data_start": spec.data_start.isoformat(),
        "data_end": spec.data_end.isoformat(),
        "split.name": spec.split.name,
        "split.version": spec.split.version,
        "split.identifier": spec.split.identifier,
        "split.digest": spec.split.digest,
        "model.name": spec.model.name,
        "model.version": spec.model.version,
        "model.identifier": spec.model.identifier,
        "model.digest": spec.model.digest,
        "model.type": spec.model.model_type,
    }
    if git_revision is not None:
        parameters["git.commit"] = git_revision
    model_manifest = spec.model.to_manifest()
    hyperparameters = model_manifest["hyperparameters"]
    if not isinstance(hyperparameters, Mapping):  # pragma: no cover - contract invariant
        raise TypeError("Model manifest hyperparameters must be a mapping.")
    parameters.update(
        {
            f"model.hyperparameter.{name}": _parameter_value(value)
            for name, value in sorted(hyperparameters.items())
        }
    )
    parameters.update(
        {f"random_seed.{name}": seed for name, seed in sorted(spec.random_seeds.items())}
    )
    return parameters


def _dataset_parameters(summary: DatasetSummary) -> dict[str, bool | int | float | str]:
    parameters: dict[str, bool | int | float | str] = {}
    for split_name in ("train", "validation", "test"):
        split = getattr(summary, split_name)
        prefix = f"dataset.{split_name}"
        parameters[f"{prefix}.rows"] = split.rows
        parameters[f"{prefix}.ticker_count"] = split.ticker_count
        parameters[f"{prefix}.start_date"] = (
            split.start_date.isoformat() if split.start_date is not None else ""
        )
        parameters[f"{prefix}.end_date"] = (
            split.end_date.isoformat() if split.end_date is not None else ""
        )
    return parameters


def _dataset_metrics(summary: DatasetSummary) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name in ("train", "validation", "test"):
        split = getattr(summary, split_name)
        if split.class_prevalence is not None:
            metrics[f"dataset.{split_name}.class_prevalence"] = float(split.class_prevalence)
    return metrics


def _validate_dataset_summary(spec: ExperimentSpec, summary: DatasetSummary) -> None:
    declared_ranges = {
        "train": (spec.split.train_start, spec.split.train_end),
        "validation": (
            spec.split.validation_start,
            spec.split.validation_end,
        ),
        "test": (spec.split.test_start, spec.split.test_end),
    }
    universe_size = len(spec.universe.tickers)

    for split_name, (declared_start, declared_end) in declared_ranges.items():
        observed = getattr(summary, split_name)
        if observed.ticker_count > universe_size:
            raise ValueError(
                f"Dataset {split_name} ticker count must not exceed the experiment universe size."
            )
        observed_start = observed.start_date
        observed_end = observed.end_date
        if observed_start is None or observed_end is None:
            continue
        if observed_start < declared_start or observed_end > declared_end:
            raise ValueError(
                f"Dataset {split_name} date range must fall within the declared temporal split."
            )
