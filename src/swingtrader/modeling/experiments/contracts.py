"""Immutable contracts for reproducible model-experiment semantics.

Feature and target contracts describe how model inputs and labels are computed.
This module composes those contracts with the remaining choices that define an
experiment: the resolved universe, temporal split, data cutoff, model
configuration, selected task, and random seeds.

The specifications are independent of MLflow. They can therefore be validated,
serialized, and assigned deterministic digests before any dataset is built or
model is fitted. Runtime provenance such as the Git revision is recorded by the
tracking adapter because it describes an execution, not the static experiment
configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType

from swingtrader.data.features.contracts import FeatureSetSpec
from swingtrader.modeling.datasets.contracts import SupervisedTaskSpec, TargetSetSpec
from swingtrader.modeling.datasets.specifications import TemporalDatasetSpec, UniverseSpec


@dataclass(frozen=True, slots=True)
class TemporalSplitSpec:
    """Declare non-overlapping calendar ranges for train, validation, and test.

    This contract records split semantics only. Applying the ranges and purging
    rows whose target horizon crosses a boundary belongs to the later temporal
    splitting implementation.
    """

    name: str
    version: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def __post_init__(self) -> None:
        _require_text(self.name, field_name="Temporal split name")
        _require_text(self.version, field_name="Temporal split version")

        ranges = (
            ("train", self.train_start, self.train_end),
            ("validation", self.validation_start, self.validation_end),
            ("test", self.test_start, self.test_end),
        )
        for range_name, start, end in ranges:
            _require_date(start, field_name=f"{range_name.title()} split start")
            _require_date(end, field_name=f"{range_name.title()} split end")
            if start > end:
                raise ValueError(f"{range_name.title()} split start must not follow its end.")

        if self.train_end >= self.validation_start:
            raise ValueError("Training and validation ranges must not overlap.")
        if self.validation_end >= self.test_start:
            raise ValueError("Validation and test ranges must not overlap.")

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

    def to_manifest(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_manifest())


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describe a versioned model implementation and its hyperparameters.

    ``model_type`` records the import path of the concrete model implementation
    so a run can be traced back to the code that produced it. ``hyperparameters``
    must contain only JSON-compatible values (booleans, integers, finite floats,
    strings, and nested mappings, lists, or tuples); they are deep-frozen into
    read-only, deterministic structures so the manifest digest is stable and the
    specification cannot be mutated after construction.
    """

    name: str
    version: str
    model_type: str
    hyperparameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, field_name="Model name")
        _require_text(self.version, field_name="Model version")
        _require_text(self.model_type, field_name="Model type")

        if not isinstance(self.hyperparameters, Mapping):
            raise TypeError("Model hyperparameters must be a mapping.")
        frozen = _freeze_value(self.hyperparameters, path="Model hyperparameters")
        if not isinstance(frozen, Mapping):  # pragma: no cover - checked above
            raise TypeError("Model hyperparameters must be a mapping.")
        object.__setattr__(self, "hyperparameters", frozen)

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "model_type": self.model_type,
            "hyperparameters": _manifest_value(self.hyperparameters),
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_manifest())


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Compose all static choices required to identify one model experiment.

    The specification binds the feature and target contracts to a resolved
    universe, temporal split, model configuration, and random seeds, producing a
    single deterministic manifest and digest for experiment identity.

    Construction enforces the invariants that keep the manifest coherent: the
    feature set, target set, task, universe, and cutoff must form a valid temporal
    dataset specification, ``data_cutoff`` must not precede the end of the declared
    test range, and at least one random seed must be provided as a non-negative
    integer. Seeds are frozen into a read-only mapping. Runtime
    provenance such as the Git revision is deliberately excluded because it
    describes an execution rather than the static experiment configuration.
    """

    name: str
    version: str
    feature_set: FeatureSetSpec
    target_set: TargetSetSpec
    task: SupervisedTaskSpec
    universe: UniverseSpec
    data_cutoff: date
    split: TemporalSplitSpec
    model: ModelSpec
    random_seeds: Mapping[str, int]

    def __post_init__(self) -> None:
        _require_text(self.name, field_name="Experiment name")
        _require_text(self.version, field_name="Experiment version")
        _ = TemporalDatasetSpec(
            feature_set=self.feature_set,
            target_set=self.target_set,
            task=self.task,
            universe=self.universe,
            data_cutoff=self.data_cutoff,
        )

        if self.data_cutoff < self.split.test_end:
            raise ValueError("Data cutoff must not precede the end of the declared test range.")

        if not isinstance(self.random_seeds, Mapping):
            raise TypeError("Random seeds must be provided as a mapping.")
        seeds = dict(self.random_seeds)
        if not seeds:
            raise ValueError("An experiment must declare at least one random seed.")
        for name, seed in seeds.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Random-seed names must be non-empty strings.")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("Random seeds must be non-negative integers.")
        object.__setattr__(self, "random_seeds", MappingProxyType(seeds))

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def dataset_spec(self) -> TemporalDatasetSpec:
        """Return the lower-level unsplit dataset specification."""
        return TemporalDatasetSpec(
            feature_set=self.feature_set,
            target_set=self.target_set,
            task=self.task,
            universe=self.universe,
            data_cutoff=self.data_cutoff,
        )

    def to_manifest(self) -> dict[str, object]:
        """Return the complete deterministic experiment configuration."""
        feature_manifest = self.feature_set.to_manifest()
        target_manifest = self.target_set.to_manifest()
        universe_manifest = self.universe.to_manifest()
        split_manifest = self.split.to_manifest()
        model_manifest = self.model.to_manifest()

        return {
            "manifest_schema_version": 1,
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "feature_set": {**feature_manifest, "digest": self.feature_set.digest},
            "target_set": {**target_manifest, "digest": self.target_set.digest},
            "task": self.task.to_manifest(),
            "universe": {**universe_manifest, "digest": self.universe.digest},
            "data_cutoff": self.data_cutoff.isoformat(),
            "split": {**split_manifest, "digest": self.split.digest},
            "model": {**model_manifest, "digest": self.model.digest},
            "random_seeds": dict(sorted(self.random_seeds.items())),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the experiment manifest with deterministic key ordering."""
        return json.dumps(self.to_manifest(), indent=indent, sort_keys=True)

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical experiment manifest."""
        return _digest(self.to_manifest())


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_date(value: date, *, field_name: str) -> None:
    if type(value) is not date:
        raise TypeError(f"{field_name} must be a datetime.date.")


def _freeze_value(value: object, *, path: str) -> object:
    """Validate and freeze one JSON-compatible configuration value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinite values.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{path} mapping keys must be non-empty strings.")
            frozen[key] = _freeze_value(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, path=f"{path}[{position}]") for position, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains unsupported value {value!r}; use JSON-compatible "
        "scalars, mappings, lists, or tuples."
    )


def _manifest_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _manifest_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_manifest_value(item) for item in value]
    return value


def _canonical_json(manifest: Mapping[str, object]) -> str:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


def _digest(manifest: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
