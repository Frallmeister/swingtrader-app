"""Expanding temporal folds confined to an existing outer training split."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from swingtrader.modeling.datasets.temporal import (
    TARGET_END_DATE_COLUMN,
    TemporalDatasetBundle,
)
from swingtrader.modeling.experiments.splitting import TemporalSplitResult


@dataclass(frozen=True, slots=True)
class TemporalCrossValidationSpec:
    """Configure deterministic expanding folds inside the outer train split.

    ``validation_sessions`` counts consecutive global signal dates in each
    held-out validation period. ``minimum_train_sessions`` is enforced against
    the distinct training dates that remain after target-horizon purging.
    """

    n_folds: int
    validation_sessions: int
    minimum_train_sessions: int

    def __post_init__(self) -> None:
        _require_positive_int(self.n_folds, field_name="Cross-validation fold count")
        _require_positive_int(
            self.validation_sessions,
            field_name="Cross-validation validation sessions",
        )
        _require_positive_int(
            self.minimum_train_sessions,
            field_name="Cross-validation minimum train sessions",
        )

    def to_manifest(self) -> dict[str, int]:
        """Return the compact cross-validation configuration."""
        return {
            "n_folds": self.n_folds,
            "validation_sessions": self.validation_sessions,
            "minimum_train_sessions": self.minimum_train_sessions,
        }


@dataclass(frozen=True, slots=True)
class TemporalFold:
    """Retain one purged expanding train/validation index pair.

    The date fields are inclusive candidate partition boundaries used for
    target-end containment. Because rows whose labels cross an end boundary
    are removed, ``train_end`` and ``validation_end`` need not occur among the
    retained signal dates. Both index arrays refer to the original dataset
    bundle and are subsets of the assigned outer training rows.
    """

    number: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date

    def __post_init__(self) -> None:
        _require_positive_int(self.number, field_name="Temporal fold number")
        train = _freeze_indices(self.train_indices, name="Fold train indices")
        validation = _freeze_indices(
            self.validation_indices,
            name="Fold validation indices",
        )
        if np.intersect1d(train, validation).size:
            raise ValueError("Fold train and validation indices must not overlap.")
        for field_name, value in (
            ("Fold train start", self.train_start),
            ("Fold train end", self.train_end),
            ("Fold validation start", self.validation_start),
            ("Fold validation end", self.validation_end),
        ):
            if type(value) is not date:
                raise TypeError(f"{field_name} must be a datetime.date.")
        if self.train_start > self.train_end:
            raise ValueError("Fold train start must not follow fold train end.")
        if self.validation_start > self.validation_end:
            raise ValueError("Fold validation start must not follow fold validation end.")
        if self.train_end >= self.validation_start:
            raise ValueError("Fold validation must start strictly after fold training.")
        object.__setattr__(self, "train_indices", train)
        object.__setattr__(self, "validation_indices", validation)


def build_expanding_temporal_folds(
    dataset: TemporalDatasetBundle,
    outer_split: TemporalSplitResult,
    *,
    spec: TemporalCrossValidationSpec,
) -> tuple[TemporalFold, ...]:
    """Build end-aligned expanding folds using only assigned outer-train rows.

    Global signal dates define each candidate partition. The existing per-row
    ``target_end_date`` metadata then applies the same inclusive containment
    rule used by the outer splitter: retained targets must resolve no later than
    their partition end. Outer validation and locked-test indices are never read.
    """
    if not isinstance(dataset, TemporalDatasetBundle):
        raise TypeError("Temporal cross-validation requires a TemporalDatasetBundle.")
    if not isinstance(outer_split, TemporalSplitResult):
        raise TypeError("Temporal cross-validation requires a TemporalSplitResult.")
    if not isinstance(spec, TemporalCrossValidationSpec):
        raise TypeError("Temporal cross-validation requires a TemporalCrossValidationSpec.")
    if outer_split.manifest.dataset_manifest_digest != dataset.manifest.digest:
        raise ValueError("Temporal split result does not belong to the supplied dataset bundle.")

    outer_train_indices = outer_split.indices("train")
    if not len(outer_train_indices):
        raise ValueError("Outer train split must contain rows for temporal cross-validation.")

    signal_dates = pd.DatetimeIndex(dataset.samples.index.get_level_values("trading_date"))
    target_end_dates = pd.DatetimeIndex(dataset.samples[TARGET_END_DATE_COLUMN])
    outer_signal_dates = signal_dates.take(outer_train_indices)
    outer_target_end_dates = target_end_dates.take(outer_train_indices)
    observed_dates = pd.DatetimeIndex(outer_signal_dates.unique()).sort_values()

    reserved_validation_sessions = spec.n_folds * spec.validation_sessions
    initial_train_sessions = len(observed_dates) - reserved_validation_sessions
    if initial_train_sessions < spec.minimum_train_sessions:
        required = spec.minimum_train_sessions + reserved_validation_sessions
        raise ValueError(
            "Outer train split has too few global dates for temporal cross-validation: "
            f"observed {len(observed_dates)}, require at least {required}."
        )

    folds: list[TemporalFold] = []
    for offset in range(spec.n_folds):
        train_stop = initial_train_sessions + offset * spec.validation_sessions
        validation_stop = train_stop + spec.validation_sessions
        train_dates = observed_dates[:train_stop]
        validation_dates = observed_dates[train_stop:validation_stop]
        train_end = train_dates[-1]
        validation_start = validation_dates[0]
        validation_end = validation_dates[-1]

        train_mask = (outer_signal_dates <= train_end) & (outer_target_end_dates <= train_end)
        validation_mask = (
            (outer_signal_dates >= validation_start)
            & (outer_signal_dates <= validation_end)
            & (outer_target_end_dates <= validation_end)
        )
        train_indices = outer_train_indices[np.asarray(train_mask, dtype=bool)]
        validation_indices = outer_train_indices[np.asarray(validation_mask, dtype=bool)]
        if not len(train_indices):
            raise ValueError(f"Temporal fold {offset + 1} has no training rows after purging.")
        if not len(validation_indices):
            raise ValueError(f"Temporal fold {offset + 1} has no validation rows after purging.")

        retained_train_dates = signal_dates.take(train_indices).unique()
        if len(retained_train_dates) < spec.minimum_train_sessions:
            raise ValueError(
                f"Temporal fold {offset + 1} retains {len(retained_train_dates)} training "
                "sessions after target-horizon purging, fewer than "
                f"minimum_train_sessions={spec.minimum_train_sessions}."
            )

        folds.append(
            TemporalFold(
                number=offset + 1,
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_start=train_dates[0].date(),
                train_end=train_end.date(),
                validation_start=validation_start.date(),
                validation_end=validation_end.date(),
            )
        )
    return tuple(folds)


def _freeze_indices(indices: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if values.size < 1:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"{name} must contain integers.")
    frozen = values.astype("int64", copy=True)
    if (frozen < 0).any():
        raise ValueError(f"{name} must not contain negative positions.")
    if len(np.unique(frozen)) != len(frozen):
        raise ValueError(f"{name} must contain unique positions.")
    if len(frozen) > 1 and (np.diff(frozen) <= 0).any():
        raise ValueError(f"{name} must be strictly increasing.")
    frozen.setflags(write=False)
    return frozen


def _require_positive_int(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be positive.")
