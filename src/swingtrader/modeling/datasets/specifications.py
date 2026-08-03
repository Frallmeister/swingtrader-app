"""Specifications that identify one unsplit temporal modeling dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

from swingtrader.data.features.contracts import FeatureSetSpec
from swingtrader.modeling.datasets.contracts import SupervisedTaskSpec, TargetSetSpec


@dataclass(frozen=True, slots=True)
class UniverseSpec:
    """Identify one resolved, versioned provider/ticker universe.

    Concrete ticker membership is stored rather than a path to mutable
    configuration. Membership order has no modeling meaning, so tickers are
    normalized and sorted before manifest generation.
    """

    name: str
    version: str
    provider: str
    tickers: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.name, field_name="Universe name")
        _require_text(self.version, field_name="Universe version")
        _require_text(self.provider, field_name="Universe provider")

        if isinstance(self.tickers, str):
            raise TypeError("Universe tickers must be an iterable of ticker strings.")
        try:
            tickers = tuple(self.tickers)
        except TypeError as exc:
            raise TypeError("Universe tickers must be an iterable of ticker strings.") from exc
        if not tickers:
            raise ValueError("A universe must contain at least one ticker.")
        if any(not isinstance(ticker, str) or not ticker.strip() for ticker in tickers):
            raise ValueError("Universe tickers must be non-empty strings.")
        tickers = tuple(sorted(ticker.strip() for ticker in tickers))
        if len(tickers) != len(set(tickers)):
            raise ValueError("Universe tickers must be unique.")
        object.__setattr__(self, "tickers", tickers)

    @property
    def identifier(self) -> str:
        """Return the stable universe name and version identifier."""
        return f"{self.name}:{self.version}"

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic resolved-universe description."""
        return {
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "provider": self.provider,
            "tickers": list(self.tickers),
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical universe manifest."""
        return _digest(self.to_manifest())


@dataclass(frozen=True, slots=True)
class TemporalDatasetSpec:
    """Bind contracts and source scope for one unsplit temporal dataset."""

    feature_set: FeatureSetSpec
    target_set: TargetSetSpec
    task: SupervisedTaskSpec
    universe: UniverseSpec
    data_start: date
    data_end: date

    def __post_init__(self) -> None:
        self.task.validate_target_set(self.target_set)
        _require_date(self.data_start, field_name="Data start")
        _require_date(self.data_end, field_name="Data end")
        if self.data_start > self.data_end:
            raise ValueError("Data start must not follow data end.")
        if self.task.horizon_sessions is None:
            raise ValueError("Temporal dataset tasks must declare horizon_sessions.")
        if self.task.horizon_sessions > self.target_set.maximum_horizon_sessions:
            raise ValueError("Task horizon must not exceed the target-set maximum horizon.")

    def to_manifest(self) -> dict[str, object]:
        """Return the complete deterministic dataset specification."""
        feature_manifest = self.feature_set.to_manifest()
        target_manifest = self.target_set.to_manifest()
        universe_manifest = self.universe.to_manifest()
        return {
            "manifest_schema_version": 1,
            "feature_set": {**feature_manifest, "digest": self.feature_set.digest},
            "target_set": {**target_manifest, "digest": self.target_set.digest},
            "task": self.task.to_manifest(),
            "universe": {**universe_manifest, "digest": self.universe.digest},
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the dataset specification with deterministic key ordering."""
        return json.dumps(self.to_manifest(), indent=indent, sort_keys=True)

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical dataset specification."""
        return _digest(self.to_manifest())


def _require_text(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_date(value: date, *, field_name: str) -> None:
    if type(value) is not date:
        raise TypeError(f"{field_name} must be a datetime.date.")


def _digest(manifest: dict[str, object]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
