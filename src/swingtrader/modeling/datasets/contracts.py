"""Immutable contracts for versioned modeling target sets."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import pandas as pd

type TargetParameter = bool | int | float | str | tuple[object, ...]
type TargetBuilder = Callable[..., pd.DataFrame]
type TaskType = Literal["classification", "regression"]


@dataclass(frozen=True, slots=True)
class TargetFamilySpec:
    """Describe one executable target family and its declared schema.

    The builder receives a DataFrame as its first argument and the configured
    parameters as keyword arguments. Required input columns, produced output
    columns, and the maximum future horizon are recorded for validation and
    deterministic manifest generation.
    """

    name: str
    builder: TargetBuilder = field(repr=False, compare=False)
    parameters: Mapping[str, TargetParameter] = field(default_factory=dict)
    required_columns: frozenset[str] = frozenset()
    output_columns: tuple[str, ...] = ()
    maximum_horizon_sessions: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target family name must not be empty.")
        if self.maximum_horizon_sessions < 1:
            raise ValueError("Target family maximum horizon must be at least one session.")
        output_columns = tuple(self.output_columns)
        if not output_columns:
            raise ValueError(f"Target family {self.name!r} must declare output columns.")
        if len(output_columns) != len(set(output_columns)):
            raise ValueError(f"Target family {self.name!r} contains duplicate output columns.")
        object.__setattr__(self, "output_columns", output_columns)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "required_columns", frozenset(self.required_columns))
        signature = inspect.signature(self.builder)
        builder_parameters = tuple(signature.parameters.values())[1:]
        configurable_kinds = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        configurable_parameters = {
            parameter.name: parameter
            for parameter in builder_parameters
            if parameter.kind in configurable_kinds
        }
        unknown_parameters = set(self.parameters).difference(configurable_parameters)
        if unknown_parameters:
            names = ", ".join(sorted(unknown_parameters))
            raise ValueError(f"Unknown parameters for target family {self.name!r}: {names}.")
        missing_parameters = {
            name
            for name, parameter in configurable_parameters.items()
            if parameter.default is inspect.Parameter.empty and name not in self.parameters
        }
        if missing_parameters:
            names = ", ".join(sorted(missing_parameters))
            raise ValueError(
                f"Missing required parameters for target family {self.name!r}: {names}."
            )

    @property
    def builder_path(self) -> str:
        """Return the import path of the configured builder."""
        return f"{self.builder.__module__}.{self.builder.__qualname__}"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply this family with its declared parameters."""
        return self.builder(data, **self.parameters)

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-serializable family description."""
        return {
            "name": self.name,
            "builder": self.builder_path,
            "parameters": {
                key: _json_value(value) for key, value in sorted(self.parameters.items())
            },
            "required_columns": sorted(self.required_columns),
            "output_columns": list(self.output_columns),
            "maximum_horizon_sessions": self.maximum_horizon_sessions,
        }


@dataclass(frozen=True, slots=True)
class TargetSetSpec:
    """Declare an ordered, versioned collection of target families.

    Families execute in declaration order, allowing later families to consume
    outputs produced by earlier families.
    """

    name: str
    version: str
    families: tuple[TargetFamilySpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Target set name must not be empty.")
        if not self.version:
            raise ValueError("Target set version must not be empty.")
        families = tuple(self.families)
        object.__setattr__(self, "families", families)
        if not families:
            raise ValueError("A target set must contain at least one family.")
        if len(self.family_names) != len(set(self.family_names)):
            raise ValueError("Target family names must be unique within a target set.")
        if len(self.target_columns) != len(set(self.target_columns)):
            raise ValueError("Target output columns must be unique across a target set.")

    @property
    def identifier(self) -> str:
        return f"{self.name}:{self.version}"

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(family.name for family in self.families)

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(column for family in self.families for column in family.output_columns)

    @property
    def maximum_horizon_sessions(self) -> int:
        """Return the greatest future horizon required by any target family."""
        return max(family.maximum_horizon_sessions for family in self.families)

    def to_manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "target_columns": list(self.target_columns),
            "maximum_horizon_sessions": self.maximum_horizon_sessions,
            "families": [family.to_manifest() for family in self.families],
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical target-set manifest."""
        payload = json.dumps(self.to_manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SupervisedTaskSpec:
    """Identify one model task against one versioned target column."""

    name: str
    target_set_name: str
    target_set_version: str
    target_column: str
    task_type: TaskType

    def __post_init__(self) -> None:
        if not all((self.name, self.target_set_name, self.target_set_version, self.target_column)):
            raise ValueError("Supervised task identifiers must not be empty.")
        if self.task_type not in {"classification", "regression"}:
            raise ValueError("Task type must be 'classification' or 'regression'.")

    def validate_target_set(self, target_set: TargetSetSpec) -> None:
        """Validate that the referenced target set and target column exist."""
        if (self.target_set_name, self.target_set_version) != (
            target_set.name,
            target_set.version,
        ):
            raise ValueError("Supervised task references a different target set.")
        if self.target_column not in target_set.target_columns:
            raise ValueError(f"Unknown target column: {self.target_column}.")

    def to_manifest(self) -> dict[str, str]:
        """Return a JSON-serializable supervised-task description."""
        return {
            "name": self.name,
            "target_set_name": self.target_set_name,
            "target_set_version": self.target_set_version,
            "target_column": self.target_column,
            "task_type": self.task_type,
        }


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
