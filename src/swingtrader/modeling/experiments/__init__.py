"""Experiment specifications and optional MLflow tracking helpers."""

from swingtrader.modeling.experiments.contracts import (
    ExperimentSpec,
    ModelSpec,
    TemporalSplitSpec,
    UniverseSpec,
    resolve_model_feature_columns,
)
from swingtrader.modeling.experiments.cross_validation import (
    TemporalCrossValidationSpec,
    TemporalFold,
    build_expanding_temporal_folds,
)
from swingtrader.modeling.experiments.splitting import (
    EMBARGO_REASON,
    OUTSIDE_SPLIT_RANGES_REASON,
    SPLIT_COLUMN,
    SPLIT_EXCLUSION_REASON_COLUMN,
    TARGET_END_AFTER_SPLIT_REASON,
    FixedTemporalSplitter,
    TemporalSplitManifest,
    TemporalSplitResult,
    TemporalSplitSummary,
    split_temporal_dataset,
)
from swingtrader.modeling.experiments.tracking import (
    DatasetSplitSummary,
    DatasetSummary,
    ExperimentRun,
    local_tracking_uri,
    resolve_git_revision,
    start_experiment_run,
)

__all__ = [
    "DatasetSplitSummary",
    "DatasetSummary",
    "EMBARGO_REASON",
    "ExperimentRun",
    "ExperimentSpec",
    "FixedTemporalSplitter",
    "ModelSpec",
    "OUTSIDE_SPLIT_RANGES_REASON",
    "SPLIT_COLUMN",
    "SPLIT_EXCLUSION_REASON_COLUMN",
    "TARGET_END_AFTER_SPLIT_REASON",
    "TemporalCrossValidationSpec",
    "TemporalFold",
    "TemporalSplitManifest",
    "TemporalSplitResult",
    "TemporalSplitSpec",
    "TemporalSplitSummary",
    "UniverseSpec",
    "build_expanding_temporal_folds",
    "local_tracking_uri",
    "resolve_git_revision",
    "resolve_model_feature_columns",
    "split_temporal_dataset",
    "start_experiment_run",
]
