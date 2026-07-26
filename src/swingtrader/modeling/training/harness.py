"""Reusable baseline fitting, split evaluation, and artifact logging harness."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from swingtrader.modeling.datasets.tabular import to_tabular_dataset
from swingtrader.modeling.datasets.temporal import TemporalDatasetBundle
from swingtrader.modeling.experiments.contracts import (
    ExperimentSpec,
    TemporalSplitName,
    resolve_model_feature_columns,
)
from swingtrader.modeling.experiments.cross_validation import (
    TemporalCrossValidationSpec,
    build_expanding_temporal_folds,
)
from swingtrader.modeling.experiments.splitting import TemporalSplitResult
from swingtrader.modeling.experiments.tracking import ExperimentRun
from swingtrader.modeling.training.baselines import (
    LOGISTIC_REGRESSION_MODEL_TYPE,
    BaselineModelArtifact,
    fit_baseline_model,
)
from swingtrader.modeling.training.contracts import EvaluationConfig, build_prediction_frame
from swingtrader.modeling.training.evaluation import EvaluationReport, evaluate_predictions
from swingtrader.modeling.training.reporting import write_evaluation_artifacts

TEMPORAL_CV_RESULT_COLUMNS = (
    "fold",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "train_rows",
    "validation_rows",
    "train_precision",
    "validation_precision",
    "train_recall",
    "validation_recall",
    "train_roc_auc",
    "validation_roc_auc",
)


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """Retain the fitted baseline and independently evaluated temporal splits."""

    model: BaselineModelArtifact
    reports: Mapping[str, EvaluationReport]

    def __post_init__(self) -> None:
        reports = dict(self.reports)
        if not reports:
            raise ValueError("A baseline experiment result must contain an evaluation report.")
        object.__setattr__(self, "reports", MappingProxyType(reports))


def run_baseline_experiment(
    bundle: TemporalDatasetBundle,
    split_result: TemporalSplitResult,
    experiment: ExperimentSpec,
    *,
    ranking_return_column: str | None = None,
    evaluation_config: EvaluationConfig | None = None,
    include_locked_test: bool = False,
    run: ExperimentRun | None = None,
    artifact_directory: str | Path | None = None,
) -> BaselineExperimentResult:
    """Fit on train and evaluate validation, with explicit locked-test opt-in.

    The function never fits preprocessing or model state on validation or test
    rows. When ``include_locked_test`` is false, test indices are not read. A
    supplied evaluation seed must match the seed declared by the experiment.
    """
    _validate_inputs(bundle, split_result, experiment)
    if not isinstance(include_locked_test, bool):
        raise TypeError("include_locked_test must be a Boolean.")
    evaluation_seed = _resolve_seed(
        experiment.random_seeds,
        preferred=("evaluation", "sampling"),
    )
    if evaluation_config is None:
        config = EvaluationConfig(random_seed=evaluation_seed)
    else:
        if not isinstance(evaluation_config, EvaluationConfig):
            raise TypeError("evaluation_config must be an EvaluationConfig.")
        if evaluation_config.random_seed != evaluation_seed:
            raise ValueError(
                "EvaluationConfig.random_seed must match the experiment evaluation seed "
                f"({evaluation_seed})."
            )
        config = evaluation_config

    tabular = to_tabular_dataset(bundle)
    train_positions = split_result.indices("train")
    train_features = tabular.X.iloc[train_positions]
    train_target = tabular.y.iloc[train_positions]
    model = fit_baseline_model(
        experiment.model,
        features=train_features,
        target=train_target,
        seed=_resolve_seed(experiment.random_seeds, preferred=("model",)),
    )

    split_names: list[TemporalSplitName] = ["validation"]
    if include_locked_test:
        split_names.append("test")
    reports: dict[str, EvaluationReport] = {}
    for split_name in split_names:
        positions = split_result.indices(split_name)
        features = tabular.X.iloc[positions]
        target = tabular.y.iloc[positions]
        ranking_return = _ranking_return(
            bundle,
            positions=positions,
            column=ranking_return_column,
        )
        scores = model.predict_scores(features)
        predictions = build_prediction_frame(
            target=target,
            score=scores,
            split=split_name,
            classification_threshold=config.classification_threshold,
            ranking_return=ranking_return,
        )
        reports[split_name] = evaluate_predictions(
            predictions,
            features=features,
            config=config,
            ranking_return_column=ranking_return_column,
        )

    result = BaselineExperimentResult(model=model, reports=reports)
    if artifact_directory is not None:
        write_baseline_artifacts(result, artifact_directory)
    if run is not None:
        _log_result(run, result)
    return result


def run_baseline_cross_validation(
    bundle: TemporalDatasetBundle,
    split_result: TemporalSplitResult,
    experiment: ExperimentSpec,
    cv_spec: TemporalCrossValidationSpec,
    *,
    evaluation_config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Evaluate one logistic model across expanding folds inside outer train.

    Each fold fits a fresh preprocessor and estimator on its purged training
    rows, then scores those rows and the later held-out validation rows with the
    same model and classification threshold. Outer validation and locked-test
    indices are never requested.
    """
    _validate_inputs(bundle, split_result, experiment)
    if experiment.model.model_type != LOGISTIC_REGRESSION_MODEL_TYPE:
        raise ValueError(
            "Temporal cross-validation currently supports only LOGISTIC_REGRESSION_MODEL_TYPE."
        )
    evaluation_seed = _resolve_seed(
        experiment.random_seeds,
        preferred=("evaluation", "sampling"),
    )
    if evaluation_config is None:
        config = EvaluationConfig(random_seed=evaluation_seed)
    else:
        if not isinstance(evaluation_config, EvaluationConfig):
            raise TypeError("evaluation_config must be an EvaluationConfig.")
        if evaluation_config.random_seed != evaluation_seed:
            raise ValueError(
                "EvaluationConfig.random_seed must match the experiment evaluation seed "
                f"({evaluation_seed})."
            )
        config = evaluation_config

    available_columns = tuple(bundle.features.columns)
    feature_columns = resolve_model_feature_columns(experiment.model, available_columns)
    target = bundle.targets[experiment.task.target_column]
    model_seed = _resolve_seed(experiment.random_seeds, preferred=("model",))
    folds = build_expanding_temporal_folds(bundle, split_result, spec=cv_spec)
    rows: list[dict[str, object]] = []
    for fold in folds:
        train_features = bundle.features.iloc[fold.train_indices].loc[:, list(feature_columns)]
        validation_features = bundle.features.iloc[fold.validation_indices].loc[
            :, list(feature_columns)
        ]
        train_target = target.iloc[fold.train_indices]
        validation_target = target.iloc[fold.validation_indices]
        model = fit_baseline_model(
            experiment.model,
            features=train_features,
            target=train_target,
            seed=model_seed,
        )
        train_metrics = _compact_classification_metrics(
            train_target,
            model.predict_scores(train_features),
            threshold=config.classification_threshold,
        )
        validation_metrics = _compact_classification_metrics(
            validation_target,
            model.predict_scores(validation_features),
            threshold=config.classification_threshold,
        )
        rows.append(
            {
                "fold": fold.number,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_rows": len(fold.train_indices),
                "validation_rows": len(fold.validation_indices),
                "train_precision": train_metrics["precision"],
                "validation_precision": validation_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "validation_recall": validation_metrics["recall"],
                "train_roc_auc": train_metrics["roc_auc"],
                "validation_roc_auc": validation_metrics["roc_auc"],
            }
        )
    return pd.DataFrame(rows, columns=TEMPORAL_CV_RESULT_COLUMNS)


def write_baseline_artifacts(
    result: BaselineExperimentResult,
    directory: str | Path,
) -> tuple[Path, ...]:
    """Write the fitted model manifest and all split evaluation artifacts."""
    if not isinstance(result, BaselineExperimentResult):
        raise TypeError("Artifact writing requires a BaselineExperimentResult.")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.json"
    model_path.write_text(
        json.dumps(result.model.to_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths = [model_path]
    for split, report in result.reports.items():
        paths.extend(write_evaluation_artifacts(report, root / split))
    return tuple(paths)


def _log_result(run: ExperimentRun, result: BaselineExperimentResult) -> None:
    if not isinstance(run, ExperimentRun):
        raise TypeError("run must be an ExperimentRun.")
    metrics = {
        f"{split}.{name}": value
        for split, report in result.reports.items()
        for name, value in report.aggregate_metrics.items()
        if math.isfinite(value)
    }
    if metrics:
        run.log_metrics(metrics)
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        for path in write_baseline_artifacts(result, root):
            relative_parent = path.relative_to(root).parent
            artifact_path = None if str(relative_parent) == "." else relative_parent.as_posix()
            run.log_artifact(path, artifact_path=artifact_path)


def _compact_classification_metrics(
    target: pd.Series,
    scores: pd.Series,
    *,
    threshold: float,
) -> dict[str, float]:
    if not target.index.equals(scores.index):
        raise ValueError("Cross-validation target and scores must share an index.")
    target_values = target.to_numpy(dtype="int8")
    score_values = scores.to_numpy(dtype="float64")
    predicted = (score_values >= threshold).astype("int8")
    positive_count = int(target_values.sum())
    negative_count = len(target_values) - positive_count
    return {
        "precision": float(precision_score(target_values, predicted, zero_division=0.0)),
        "recall": (
            float(recall_score(target_values, predicted, zero_division=0.0))
            if positive_count
            else math.nan
        ),
        "roc_auc": (
            float(roc_auc_score(target_values, score_values))
            if positive_count and negative_count
            else math.nan
        ),
    }


def _validate_inputs(
    bundle: TemporalDatasetBundle,
    split_result: TemporalSplitResult,
    experiment: ExperimentSpec,
) -> None:
    if not isinstance(bundle, TemporalDatasetBundle):
        raise TypeError("Baseline experiments require a TemporalDatasetBundle.")
    if not isinstance(split_result, TemporalSplitResult):
        raise TypeError("Baseline experiments require a TemporalSplitResult.")
    if not isinstance(experiment, ExperimentSpec):
        raise TypeError("Baseline experiments require an ExperimentSpec.")
    if bundle.manifest.spec.digest != experiment.dataset_spec.digest:
        raise ValueError("Temporal dataset specification does not match the experiment.")
    if split_result.manifest.dataset_manifest_digest != bundle.manifest.digest:
        raise ValueError("Temporal split result does not belong to the supplied dataset bundle.")
    if split_result.manifest.spec.digest != experiment.split.digest:
        raise ValueError("Temporal split policy does not match the experiment.")


def _ranking_return(
    bundle: TemporalDatasetBundle,
    *,
    positions: np.ndarray,
    column: str | None,
) -> pd.Series | None:
    if column is None:
        return None
    if not isinstance(column, str) or not column.strip():
        raise ValueError("ranking_return_column must be a non-empty string when provided.")
    if column not in bundle.targets.columns:
        raise ValueError(f"Ranking-return column {column!r} is not present in bundle targets.")
    return bundle.targets.iloc[positions][column]


def _resolve_seed(seeds: Mapping[str, int], *, preferred: tuple[str, ...]) -> int:
    for name in preferred:
        if name in seeds:
            return seeds[name]
    return seeds[sorted(seeds)[0]]
