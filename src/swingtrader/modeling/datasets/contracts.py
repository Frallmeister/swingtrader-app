"""Executable contracts for reproducible modeling target sets.

A :class:`TargetFamilySpec` binds one future-dependent builder to its recorded
parameters and output schema. A :class:`TargetSetSpec` composes families in
declaration order, while :class:`SupervisedTaskSpec` selects one generated
target and its resolution semantics for downstream dataset construction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from swingtrader.core.dataframe_contracts import (
    ContractParameter,
    execute_dataframe_contract,
    normalize_builder_parameters,
)

type TargetParameter = ContractParameter
type TargetBuilder = Callable[..., pd.DataFrame]
type TaskType = Literal["classification", "regression"]


@dataclass(frozen=True, slots=True)
class TargetFamilySpec:
    """Define and enforce one future-dependent target calculation.

    The builder receives a DataFrame followed by the recorded parameters as
    keyword arguments. :meth:`apply` validates required inputs, output
    collisions, index preservation, and declared outputs, then returns only
    ``output_columns`` in their declared order.
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
        if any(not isinstance(column, str) or not column for column in output_columns):
            raise ValueError("Target output columns must be non-empty strings.")

        parameters = normalize_builder_parameters(
            self.builder,
            self.parameters,
            subject=f"target family {self.name!r}",
        )
        required_columns = frozenset(self.required_columns)
        if any(not isinstance(column, str) or not column for column in required_columns):
            raise ValueError("Target required columns must be non-empty strings.")

        object.__setattr__(self, "output_columns", output_columns)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "required_columns", required_columns)

    @property
    def builder_path(self) -> str:
        """Return the import path of the configured builder."""
        return f"{self.builder.__module__}.{self.builder.__qualname__}"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the family and return its declared target columns."""
        return execute_dataframe_contract(
            data,
            builder=self.builder,
            parameters=self.parameters,
            required_columns=self.required_columns,
            output_columns=self.output_columns,
            subject=f"Target family {self.name!r}",
        )

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
    """Define and execute an ordered, versioned target set.

    Families execute in declaration order, allowing a later family to require
    a target produced by an earlier family. The returned frame contains the
    original input columns followed by exactly ``target_columns``.
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
        """Return the stable target-set name and version identifier."""
        return f"{self.name}:{self.version}"

    @property
    def family_names(self) -> tuple[str, ...]:
        """Return the family names in declared execution order."""
        return tuple(family.name for family in self.families)

    @property
    def target_columns(self) -> tuple[str, ...]:
        """Return all declared target columns in execution order."""
        return tuple(column for family in self.families for column in family.output_columns)

    @property
    def maximum_horizon_sessions(self) -> int:
        """Return the greatest future horizon required by any target family."""
        return max(family.maximum_horizon_sessions for family in self.families)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Return inputs that must exist before the target set executes."""
        produced: set[str] = set()
        source_columns: set[str] = set()
        for family in self.families:
            source_columns.update(family.required_columns.difference(produced))
            produced.update(family.output_columns)
        return tuple(sorted(source_columns))

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return an independent frame with the declared targets appended."""
        result = data.copy(deep=False)
        for family in self.families:
            family_output = family.apply(result)
            result = pd.concat((result, family_output), axis="columns")
        return result

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-serializable target-set manifest."""
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
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SupervisedTaskSpec:
    """Identify one model task and its target-resolution semantics.

    ``horizon_sessions`` identifies the selected target's own future horizon,
    which can be shorter than the maximum horizon of its target set. When
    ``target_end_date_column`` is set, that generated target column records the
    actual event or timeout date. Otherwise a dataset builder derives the fixed
    observed-session horizon from the canonical index.
    """

    name: str
    target_set_name: str
    target_set_version: str
    target_column: str
    task_type: TaskType
    horizon_sessions: int | None = None
    target_end_date_column: str | None = None

    def __post_init__(self) -> None:
        if not all((self.name, self.target_set_name, self.target_set_version, self.target_column)):
            raise ValueError("Supervised task identifiers must not be empty.")
        if self.task_type not in {"classification", "regression"}:
            raise ValueError("Task type must be 'classification' or 'regression'.")
        if self.horizon_sessions is not None:
            if isinstance(self.horizon_sessions, bool) or not isinstance(
                self.horizon_sessions, int
            ):
                raise TypeError("Task horizon must be an integer number of sessions.")
            if self.horizon_sessions < 1:
                raise ValueError("Task horizon must be at least one session.")
        if self.target_end_date_column is not None:
            if not isinstance(self.target_end_date_column, str):
                raise TypeError("Target end-date column must be a string.")
            if not self.target_end_date_column:
                raise ValueError("Target end-date column must not be empty.")
            if self.target_end_date_column == self.target_column:
                raise ValueError("Target end-date column must differ from the target column.")

    def validate_target_set(self, target_set: TargetSetSpec) -> None:
        """Validate that referenced target outputs belong to the target set."""
        if (self.target_set_name, self.target_set_version) != (
            target_set.name,
            target_set.version,
        ):
            raise ValueError("Supervised task references a different target set.")
        if self.target_column not in target_set.target_columns:
            raise ValueError(f"Unknown target column: {self.target_column}.")
        if (
            self.target_end_date_column is not None
            and self.target_end_date_column not in target_set.target_columns
        ):
            raise ValueError(f"Unknown target end-date column: {self.target_end_date_column}.")

    def to_manifest(self) -> dict[str, object]:
        """Return a JSON-serializable supervised-task description."""
        return {
            "name": self.name,
            "target_set_name": self.target_set_name,
            "target_set_version": self.target_set_version,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "horizon_sessions": self.horizon_sessions,
            "target_end_date_column": self.target_end_date_column,
        }


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
