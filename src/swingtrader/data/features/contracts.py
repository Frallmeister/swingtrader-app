"""Executable contracts for reproducible model feature sets.

A :class:`FeatureBlockSpec` binds one feature-family builder to its recorded
parameters, required inputs, and declared output schema. Applying a block
validates that contract and returns only the declared feature columns.

A :class:`FeatureSetSpec` composes blocks in declaration order. Applying a set
returns an independent copy of the input with exactly the declared features
appended in contract order. Manifests and digests are therefore derived from
the same specifications that execute the feature calculations.

Exact reproduction still requires the source revision containing the builder
implementations and the corresponding input data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from swingtrader.core.dataframe_contracts import (
    ContractParameter,
    execute_dataframe_contract,
    normalize_builder_parameters,
)

type FeatureParameter = ContractParameter
type FeatureBuilder = Callable[..., pd.DataFrame]


class HistoryRequirement(StrEnum):
    """Describe how much historical state a feature block may depend on.

    The value governs how many prior rows must be supplied for a block to
    reproduce identical results on a sliced window of data:

    * ``BOUNDED``: each output depends on a fixed-size lookback, so a
      constant warm-up prefix is sufficient (e.g. an N-period return).
    * ``EXPANDING``: outputs depend on a growing window that reaches back
      to the start of the series (e.g. recursive/expanding statistics),
      so the full available history is required for exact results.
    * ``PATH_DEPENDENT``: outputs depend on the ordered sequence of prior
      events, not just a window, so results can differ if earlier bars
      are truncated (e.g. zigzag/market-structure state).
    """

    BOUNDED = "bounded"
    EXPANDING = "expanding"
    PATH_DEPENDENT = "path_dependent"


@dataclass(frozen=True, slots=True)
class FeatureBlockSpec:
    """Define and enforce one feature-family calculation.

    The builder receives a DataFrame followed by the recorded parameters as
    keyword arguments. :meth:`apply` validates required inputs, output
    collisions, index preservation, and declared outputs, then returns only
    ``output_columns`` in their declared order.
    """

    name: str
    builder: FeatureBuilder = field(repr=False, compare=False)
    parameters: Mapping[str, FeatureParameter] = field(default_factory=dict)
    output_columns: tuple[str, ...] = ()
    required_columns: frozenset[str] = frozenset()
    history_requirement: HistoryRequirement = HistoryRequirement.BOUNDED

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature block name must not be empty.")

        output_columns = tuple(self.output_columns)
        if not output_columns:
            raise ValueError(f"Feature block {self.name!r} must declare output columns.")
        if len(output_columns) != len(set(output_columns)):
            raise ValueError(f"Feature block {self.name!r} contains duplicate output columns.")
        if any(not isinstance(column, str) or not column for column in output_columns):
            raise ValueError("Feature output columns must be non-empty strings.")

        parameters = normalize_builder_parameters(
            self.builder,
            self.parameters,
            subject=f"feature block {self.name!r}",
        )
        required_columns = frozenset(self.required_columns)
        if any(not isinstance(column, str) or not column for column in required_columns):
            raise ValueError("Feature required columns must be non-empty strings.")

        object.__setattr__(self, "output_columns", output_columns)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "required_columns", required_columns)

    @property
    def builder_path(self) -> str:
        """Return the import path of the configured builder."""
        return f"{self.builder.__module__}.{self.builder.__qualname__}"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the block and return its declared feature columns."""
        return execute_dataframe_contract(
            data,
            builder=self.builder,
            parameters=self.parameters,
            required_columns=self.required_columns,
            output_columns=self.output_columns,
            subject=f"Feature block {self.name!r}",
        )

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-serializable block description."""
        return {
            "name": self.name,
            "builder": self.builder_path,
            "parameters": {
                key: _json_value(value) for key, value in sorted(self.parameters.items())
            },
            "output_columns": list(self.output_columns),
            "required_columns": sorted(self.required_columns),
            "history_requirement": self.history_requirement.value,
        }


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    """Define and execute an ordered, versioned feature set.

    Blocks execute in declaration order, allowing a later block to require a
    feature produced by an earlier block. The returned frame contains the
    original input columns followed by exactly ``feature_columns``.
    """

    name: str
    version: str
    blocks: tuple[FeatureBlockSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Feature set name must not be empty.")
        if not self.version:
            raise ValueError("Feature set version must not be empty.")
        blocks = tuple(self.blocks)
        object.__setattr__(self, "blocks", blocks)

        if not blocks:
            raise ValueError("A feature set must contain at least one block.")
        if len(self.block_names) != len(set(self.block_names)):
            raise ValueError("Feature block names must be unique within a feature set.")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("Feature output columns must be unique across a feature set.")

    @property
    def identifier(self) -> str:
        """Return the stable feature-set name and version identifier."""
        return f"{self.name}:{self.version}"

    @property
    def block_names(self) -> tuple[str, ...]:
        """Return the block names in declared execution order."""
        return tuple(block.name for block in self.blocks)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return all declared feature columns in execution order."""
        return tuple(column for block in self.blocks for column in block.output_columns)

    @property
    def required_columns(self) -> frozenset[str]:
        """Return the union of all inputs declared by the feature blocks."""
        return frozenset(column for block in self.blocks for column in block.required_columns)

    @property
    def source_columns(self) -> tuple[str, ...]:
        """Return inputs that must exist before the feature set executes."""
        produced: set[str] = set()
        source_columns: set[str] = set()
        for block in self.blocks:
            source_columns.update(block.required_columns.difference(produced))
            produced.update(block.output_columns)
        return tuple(sorted(source_columns))

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return an independent frame with the declared features appended."""
        result = data.copy(deep=True)
        for block in self.blocks:
            block_output = block.apply(result)
            result = pd.concat((result, block_output), axis="columns")
        return result

    def select(
        self,
        *block_names: str,
        name: str,
        version: str,
    ) -> FeatureSetSpec:
        """Return a newly identified subset in the original block order."""
        requested = set(block_names)
        if not requested:
            raise ValueError("At least one feature block name is required.")

        available = set(self.block_names)
        unknown = requested.difference(available)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown feature block names: {names}.")

        return FeatureSetSpec(
            name=name,
            version=version,
            blocks=tuple(block for block in self.blocks if block.name in requested),
        )

    def to_manifest(self) -> dict[str, object]:
        """Return a deterministic, JSON-serializable feature-set manifest."""
        return {
            "name": self.name,
            "version": self.version,
            "identifier": self.identifier,
            "feature_columns": list(self.feature_columns),
            "required_columns": sorted(self.required_columns),
            "blocks": [block.to_manifest() for block in self.blocks],
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical feature-set manifest."""
        payload = json.dumps(
            self.to_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value
