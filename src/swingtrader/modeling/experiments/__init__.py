"""Experiment specifications and optional MLflow tracking helpers."""

from swingtrader.modeling.experiments.contracts import (
    ExperimentSpec,
    ModelSpec,
    TemporalSplitSpec,
    UniverseSpec,
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
    "TemporalSplitManifest",
    "TemporalSplitResult",
    "TemporalSplitSpec",
    "TemporalSplitSummary",
    "UniverseSpec",
    "local_tracking_uri",
    "resolve_git_revision",
    "split_temporal_dataset",
    "start_experiment_run",
]
