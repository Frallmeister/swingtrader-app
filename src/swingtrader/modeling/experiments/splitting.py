"""Purged fixed temporal splitting for canonical modeling datasets.

The splitter applies one experiment-level train, validation, and locked-test
calendar policy to every ticker in a canonical temporal dataset. Candidate rows
are assigned by signal date, then removed when their actual ``target_end_date``
falls after the end of that candidate split. An optional embargo removes an
additional number of global observed signal dates from the end of train and
validation after purging.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from numbers import Real

import numpy as np
import pandas as pd

from swingtrader.data.market_frame import validate_market_price_index
from swingtrader.modeling.datasets.temporal import (
    TARGET_END_DATE_COLUMN,
    TemporalDatasetBundle,
)
from swingtrader.modeling.experiments.contracts import TemporalSplitName, TemporalSplitSpec

SPLIT_COLUMN = "split"
SPLIT_EXCLUSION_REASON_COLUMN = "split_exclusion_reason"
OUTSIDE_SPLIT_RANGES_REASON = "outside_declared_ranges"
TARGET_END_AFTER_SPLIT_REASON = "target_end_after_split_end"
EMBARGO_REASON = "embargo"

_SPLIT_NAMES: tuple[TemporalSplitName, ...] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class TemporalSplitSummary:
    """Summarize one materialized split and its exclusions."""

    name: TemporalSplitName
    candidate_row_count: int
    assigned_row_count: int
    purged_row_count: int
    embargoed_row_count: int
    trading_date_count: int
    ticker_count: int
    signal_date_start: date
    signal_date_end: date
    target_end_date_start: date
    target_end_date_end: date
    class_prevalence: float | None

    def __post_init__(self) -> None:
        _validate_split_name(self.name)
        counts = (
            self.candidate_row_count,
            self.assigned_row_count,
            self.purged_row_count,
            self.embargoed_row_count,
            self.trading_date_count,
            self.ticker_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise TypeError("Temporal split summary counts must be integers.")
        if any(value < 0 for value in counts):
            raise ValueError("Temporal split summary counts must not be negative.")
        if self.candidate_row_count != (
            self.assigned_row_count + self.purged_row_count + self.embargoed_row_count
        ):
            raise ValueError("Split candidate rows must equal assigned and excluded rows.")
        if self.assigned_row_count < 1:
            raise ValueError("A temporal split summary must contain assigned rows.")
        if self.trading_date_count < 1 or self.ticker_count < 1:
            raise ValueError("An assigned split must contain dates and tickers.")
        if self.trading_date_count > self.assigned_row_count:
            raise ValueError("Split trading-date count must not exceed assigned rows.")
        if self.ticker_count > self.assigned_row_count:
            raise ValueError("Split ticker count must not exceed assigned rows.")
        for field_name, value in (
            ("Split signal-date start", self.signal_date_start),
            ("Split signal-date end", self.signal_date_end),
            ("Split target-end-date start", self.target_end_date_start),
            ("Split target-end-date end", self.target_end_date_end),
        ):
            if type(value) is not date:
                raise TypeError(f"{field_name} must be a datetime.date.")
        if self.signal_date_start > self.signal_date_end:
            raise ValueError("Split signal-date start must not follow its end.")
        if self.target_end_date_start > self.target_end_date_end:
            raise ValueError("Split target-end-date start must not follow its end.")
        if self.class_prevalence is not None:
            if isinstance(self.class_prevalence, bool) or not isinstance(
                self.class_prevalence, Real
            ):
                raise TypeError("Split class prevalence must be a real number.")
            if not math.isfinite(self.class_prevalence):
                raise ValueError("Split class prevalence must be finite.")
            if not 0.0 <= self.class_prevalence <= 1.0:
                raise ValueError("Split class prevalence must be between zero and one.")

    def to_manifest(self) -> dict[str, object]:
        """Return a JSON-serializable split summary."""
        return {
            "name": self.name,
            "candidate_row_count": self.candidate_row_count,
            "assigned_row_count": self.assigned_row_count,
            "purged_row_count": self.purged_row_count,
            "embargoed_row_count": self.embargoed_row_count,
            "trading_date_count": self.trading_date_count,
            "ticker_count": self.ticker_count,
            "signal_date_range": {
                "start": self.signal_date_start.isoformat(),
                "end": self.signal_date_end.isoformat(),
            },
            "target_end_date_range": {
                "start": self.target_end_date_start.isoformat(),
                "end": self.target_end_date_end.isoformat(),
            },
            "class_prevalence": self.class_prevalence,
        }


@dataclass(frozen=True, slots=True)
class TemporalSplitManifest:
    """Deterministic diagnostics for one split assignment."""

    spec: TemporalSplitSpec
    dataset_manifest_digest: str
    source_row_count: int
    outside_range_row_count: int
    summaries: tuple[TemporalSplitSummary, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TemporalSplitSpec):
            raise TypeError("Split manifest spec must be a TemporalSplitSpec.")
        if not isinstance(self.dataset_manifest_digest, str) or not self.dataset_manifest_digest:
            raise ValueError("Split manifest dataset digest must be a non-empty string.")
        for field_name, value in (
            ("Split manifest source row count", self.source_row_count),
            ("Split manifest outside-range row count", self.outside_range_row_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 0:
                raise ValueError(f"{field_name} must not be negative.")
        if tuple(summary.name for summary in self.summaries) != _SPLIT_NAMES:
            raise ValueError("Split manifest summaries must be ordered train, validation, test.")
        ranges = {name: (start, end) for name, start, end in self.spec.ranges}
        for summary in self.summaries:
            start, end = ranges[summary.name]
            if summary.signal_date_start < start or summary.signal_date_end > end:
                raise ValueError("Assigned signal dates must remain inside their declared range.")
            if summary.target_end_date_end > end:
                raise ValueError("Assigned target end dates must not cross their split end.")
            if summary.name == "test" and summary.embargoed_row_count:
                raise ValueError("The locked test split must not apply an end embargo.")
        accounted = self.outside_range_row_count + sum(
            summary.candidate_row_count for summary in self.summaries
        )
        if accounted != self.source_row_count:
            raise ValueError("Split manifest row counts do not account for every source row.")

    @property
    def assigned_row_count(self) -> int:
        """Return the number of rows assigned to train, validation, or test."""
        return sum(summary.assigned_row_count for summary in self.summaries)

    @property
    def purged_row_count(self) -> int:
        """Return the number of rows removed for crossing a split end."""
        return sum(summary.purged_row_count for summary in self.summaries)

    @property
    def embargoed_row_count(self) -> int:
        """Return the number of additional rows removed by the embargo."""
        return sum(summary.embargoed_row_count for summary in self.summaries)

    def summary(self, name: TemporalSplitName) -> TemporalSplitSummary:
        """Return diagnostics for one named split."""
        _validate_split_name(name)
        return next(summary for summary in self.summaries if summary.name == name)

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic JSON-serializable split manifest."""
        return {
            "manifest_schema_version": 1,
            "split_spec": {**self.spec.to_manifest(), "digest": self.spec.digest},
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "source_row_count": self.source_row_count,
            "assigned_row_count": self.assigned_row_count,
            "outside_range_row_count": self.outside_range_row_count,
            "purged_row_count": self.purged_row_count,
            "embargoed_row_count": self.embargoed_row_count,
            "splits": {summary.name: summary.to_manifest() for summary in self.summaries},
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the split manifest with deterministic key ordering."""
        return json.dumps(self.to_manifest(), indent=indent, sort_keys=True)

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical split manifest."""
        payload = json.dumps(self.to_manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalSplitResult:
    """Own split-annotated sample metadata and deterministic diagnostics.

    ``samples`` retains every row of the unsplit bundle. Assigned rows have a
    value in ``split``; excluded rows instead record why they were not assigned.
    Positional indices refer to the original bundle and are suitable for pandas,
    NumPy, and scikit-learn-style consumers.
    """

    samples: pd.DataFrame
    manifest: TemporalSplitManifest

    def __post_init__(self) -> None:
        samples = self.samples.copy(deep=True)
        validate_market_price_index(samples)
        required = {TARGET_END_DATE_COLUMN, SPLIT_COLUMN, SPLIT_EXCLUSION_REASON_COLUMN}
        missing = sorted(required.difference(samples.columns))
        if missing:
            raise ValueError(f"Split sample metadata is missing columns: {', '.join(missing)}")
        if len(samples) != self.manifest.source_row_count:
            raise ValueError("Split sample row count does not match the split manifest.")
        assigned = samples[SPLIT_COLUMN].notna()
        excluded = samples[SPLIT_EXCLUSION_REASON_COLUMN].notna()
        if (assigned == excluded).any():
            raise ValueError("Each split sample must be either assigned or excluded, but not both.")
        observed_splits = set(samples.loc[assigned, SPLIT_COLUMN].astype(str))
        if not observed_splits.issubset(_SPLIT_NAMES):
            raise ValueError("Split sample metadata contains an unknown split name.")
        expected_reasons = {
            OUTSIDE_SPLIT_RANGES_REASON,
            TARGET_END_AFTER_SPLIT_REASON,
            EMBARGO_REASON,
        }
        observed_reasons = set(samples.loc[excluded, SPLIT_EXCLUSION_REASON_COLUMN].astype(str))
        if not observed_reasons.issubset(expected_reasons):
            raise ValueError("Split sample metadata contains an unknown exclusion reason.")
        if int(assigned.sum()) != self.manifest.assigned_row_count:
            raise ValueError("Assigned sample count does not match the split manifest.")
        for summary in self.manifest.summaries:
            observed_count = int(samples[SPLIT_COLUMN].eq(summary.name).fillna(False).sum())
            if observed_count != summary.assigned_row_count:
                raise ValueError(f"{summary.name.title()} sample count does not match its summary.")
        expected_reason_counts = {
            OUTSIDE_SPLIT_RANGES_REASON: self.manifest.outside_range_row_count,
            TARGET_END_AFTER_SPLIT_REASON: self.manifest.purged_row_count,
            EMBARGO_REASON: self.manifest.embargoed_row_count,
        }
        observed_reason_counts = samples[SPLIT_EXCLUSION_REASON_COLUMN].value_counts()
        for reason, expected_count in expected_reason_counts.items():
            if int(observed_reason_counts.get(reason, 0)) != expected_count:
                raise ValueError(f"Sample count for exclusion reason {reason!r} is inconsistent.")
        object.__setattr__(self, "samples", samples)

    def indices(self, name: TemporalSplitName) -> np.ndarray:
        """Return positional indices for one assigned split."""
        _validate_split_name(name)
        mask = self.samples[SPLIT_COLUMN].eq(name).fillna(False).to_numpy(dtype=bool)
        return np.flatnonzero(mask)

    def sample_index(self, name: TemporalSplitName) -> pd.MultiIndex:
        """Return the canonical sample index for one assigned split."""
        return self.samples.index.take(self.indices(name))

    def summary(self, name: TemporalSplitName) -> TemporalSplitSummary:
        """Return diagnostics for one assigned split."""
        return self.manifest.summary(name)


@dataclass(frozen=True, slots=True)
class FixedTemporalSplitter:
    """Apply one purged fixed train/validation/test split specification."""

    spec: TemporalSplitSpec

    def __post_init__(self) -> None:
        if not isinstance(self.spec, TemporalSplitSpec):
            raise TypeError("FixedTemporalSplitter requires a TemporalSplitSpec.")

    def assign(self, dataset: TemporalDatasetBundle) -> TemporalSplitResult:
        """Materialize split assignments and diagnostics for ``dataset``."""
        return split_temporal_dataset(dataset, spec=self.spec)

    def split(
        self,
        dataset: TemporalDatasetBundle,
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield one train/validation index pair for routine model selection.

        The locked test indices are intentionally omitted. Access them explicitly
        from :meth:`assign` only when performing final evaluation.
        """
        del y, groups
        result = self.assign(dataset)
        yield result.indices("train"), result.indices("validation")

    def get_n_splits(
        self,
        dataset: TemporalDatasetBundle | None = None,
        y: object = None,
        groups: object = None,
    ) -> int:
        """Return one because the initial splitter represents one fixed holdout."""
        del dataset, y, groups
        return 1


def split_temporal_dataset(
    dataset: TemporalDatasetBundle,
    *,
    spec: TemporalSplitSpec,
) -> TemporalSplitResult:
    """Assign, purge, embargo, and summarize a canonical temporal dataset.

    Rows first become candidates through their inclusive signal-date range. A
    candidate is purged when its actual target resolution date falls after that
    split's inclusive end date. When configured, embargo then removes the final
    ``embargo_sessions`` distinct global signal dates from the surviving train
    and validation candidates. It is therefore an additional pre-boundary gap,
    not a per-ticker shift and not a post-evaluation exclusion.
    """
    if not isinstance(dataset, TemporalDatasetBundle):
        raise TypeError("Temporal splitting requires a TemporalDatasetBundle.")
    if not isinstance(spec, TemporalSplitSpec):
        raise TypeError("Temporal splitting requires a TemporalSplitSpec.")
    if spec.test_end > dataset.manifest.spec.data_cutoff:
        raise ValueError("Split test end must not exceed the temporal dataset cutoff.")

    signal_dates = pd.DatetimeIndex(dataset.samples.index.get_level_values("trading_date"))
    target_end_dates = pd.DatetimeIndex(dataset.samples[TARGET_END_DATE_COLUMN])
    samples = dataset.samples.copy(deep=True)
    assignments = pd.Series(pd.NA, index=samples.index, dtype="string")
    exclusion_reasons = pd.Series(pd.NA, index=samples.index, dtype="string")
    candidate_masks: dict[TemporalSplitName, np.ndarray] = {}
    purged_masks: dict[TemporalSplitName, np.ndarray] = {}
    embargoed_masks: dict[TemporalSplitName, np.ndarray] = {}

    any_candidate = np.zeros(len(samples), dtype=bool)
    for name, start, end in spec.ranges:
        start_timestamp = pd.Timestamp(start)
        end_timestamp = pd.Timestamp(end)
        candidate = (signal_dates >= start_timestamp) & (signal_dates <= end_timestamp)
        candidate_masks[name] = candidate
        any_candidate |= candidate

        purged = candidate & (target_end_dates > end_timestamp)
        purged_masks[name] = purged
        exclusion_reasons.iloc[np.flatnonzero(purged)] = TARGET_END_AFTER_SPLIT_REASON

        surviving = candidate & ~purged
        embargoed = np.zeros(len(samples), dtype=bool)
        if name != "test" and spec.embargo_sessions:
            observed_dates = pd.DatetimeIndex(signal_dates[surviving].unique()).sort_values()
            if len(observed_dates) <= spec.embargo_sessions:
                raise ValueError(
                    f"{name.title()} split has too few surviving signal dates for "
                    f"an embargo of {spec.embargo_sessions} sessions."
                )
            embargo_dates = observed_dates[-spec.embargo_sessions :]
            embargoed = surviving & signal_dates.isin(embargo_dates)
            exclusion_reasons.iloc[np.flatnonzero(embargoed)] = EMBARGO_REASON
        embargoed_masks[name] = embargoed

        assigned = surviving & ~embargoed
        if not assigned.any():
            raise ValueError(f"{name.title()} split contains no rows after purging and embargo.")
        assignments.iloc[np.flatnonzero(assigned)] = name

    outside = ~any_candidate
    exclusion_reasons.iloc[np.flatnonzero(outside)] = OUTSIDE_SPLIT_RANGES_REASON
    samples[SPLIT_COLUMN] = assignments
    samples[SPLIT_EXCLUSION_REASON_COLUMN] = exclusion_reasons

    summaries = tuple(
        _build_split_summary(
            name=name,
            dataset=dataset,
            assignments=assignments,
            candidate_mask=candidate_masks[name],
            purged_mask=purged_masks[name],
            embargoed_mask=embargoed_masks[name],
        )
        for name in _SPLIT_NAMES
    )
    manifest = TemporalSplitManifest(
        spec=spec,
        dataset_manifest_digest=dataset.manifest.digest,
        source_row_count=len(samples),
        outside_range_row_count=int(outside.sum()),
        summaries=summaries,
    )
    return TemporalSplitResult(samples=samples, manifest=manifest)


def _build_split_summary(
    *,
    name: TemporalSplitName,
    dataset: TemporalDatasetBundle,
    assignments: pd.Series,
    candidate_mask: np.ndarray,
    purged_mask: np.ndarray,
    embargoed_mask: np.ndarray,
) -> TemporalSplitSummary:
    assigned_mask = assignments.eq(name).fillna(False).to_numpy(dtype=bool)
    assigned_samples = dataset.samples.iloc[np.flatnonzero(assigned_mask)]
    assigned_target = dataset.targets.iloc[np.flatnonzero(assigned_mask)][
        dataset.manifest.spec.task.target_column
    ]
    signal_dates = pd.DatetimeIndex(assigned_samples.index.get_level_values("trading_date"))
    target_end_dates = pd.DatetimeIndex(assigned_samples[TARGET_END_DATE_COLUMN])
    return TemporalSplitSummary(
        name=name,
        candidate_row_count=int(candidate_mask.sum()),
        assigned_row_count=int(assigned_mask.sum()),
        purged_row_count=int(purged_mask.sum()),
        embargoed_row_count=int(embargoed_mask.sum()),
        trading_date_count=signal_dates.nunique(),
        ticker_count=assigned_samples.index.get_level_values("ticker").nunique(),
        signal_date_start=signal_dates.min().date(),
        signal_date_end=signal_dates.max().date(),
        target_end_date_start=target_end_dates.min().date(),
        target_end_date_end=target_end_dates.max().date(),
        class_prevalence=_binary_class_prevalence(
            assigned_target,
            task_type=dataset.manifest.spec.task.task_type,
        ),
    )


def _binary_class_prevalence(target: pd.Series, *, task_type: str) -> float | None:
    if task_type != "classification":
        return None
    if pd.api.types.is_bool_dtype(target.dtype):
        return float(target.astype("float64").mean())
    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
        return None
    return float(numeric.mean())


def _validate_split_name(name: str) -> None:
    if name not in _SPLIT_NAMES:
        supported = ", ".join(_SPLIT_NAMES)
        raise ValueError(f"Unknown split name {name!r}; expected one of: {supported}.")
