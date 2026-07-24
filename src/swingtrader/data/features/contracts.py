"""Versioned contracts for reproducible model feature sets.

This module defines the small vocabulary used to describe feature
computations declaratively:

* A :class:`FeatureBlockSpec` binds one feature-family builder to its
  parameters and stable input/output column schema.
* A :class:`FeatureSetSpec` composes an ordered, name-versioned
  collection of blocks into a single feature contract.

Both specifications are frozen and normalize their declared inputs into
immutable containers, preventing accidental mutation after construction.

Each specification can emit a deterministic, JSON-serializable manifest
through ``to_manifest``. :class:`FeatureSetSpec` also exposes a SHA-256
digest of its canonical manifest for compact experiment and artifact
provenance.

The manifest and digest identify the declared feature configuration.
Exact reproduction additionally requires the source revision containing
the configured builder implementations and the corresponding input data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

import pandas as pd

type FeatureParameter = bool | int | float | str | tuple[object, ...]
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
    """Declare one executable feature-family block and its stable schema.

    Invariants enforced at construction: the block name is non-empty,
    ``output_columns`` is a non-empty tuple with no duplicates, and the
    inputs are frozen (``output_columns`` to a tuple, ``parameters`` to a
    read-only mapping, ``required_columns`` to a frozenset) so the spec
    cannot be mutated after creation.
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

        # Make sure output_columns are immutable in case the caller provided e.g. a list.
        output_columns = tuple(self.output_columns)
        object.__setattr__(self, "output_columns", output_columns)

        if not output_columns:
            raise ValueError(f"Feature block {self.name!r} must declare output columns.")
        if len(output_columns) != len(set(output_columns)):
            raise ValueError(f"Feature block {self.name!r} contains duplicate output columns.")

        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "required_columns", frozenset(self.required_columns))

    @property
    def builder_path(self) -> str:
        """Return the import path of the configured builder."""
        return f"{self.builder.__module__}.{self.builder.__qualname__}"

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply this block with its declared parameters."""
        return self.builder(data, **self.parameters)

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
    """Declare an ordered, versioned collection of feature blocks.

    Invariants enforced at construction: the name and version are
    non-empty, ``blocks`` is a non-empty tuple, block names are unique,
    and every output column is unique across the whole set. Blocks retain
    their declared order, which is the order features are computed.
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

        block_names = self.block_names

        if len(block_names) != len(set(block_names)):
            raise ValueError("Feature block names must be unique within a feature set.")

        output_columns = self.feature_columns
        if len(output_columns) != len(set(output_columns)):
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
        """Return the union of source columns required by all blocks."""
        return frozenset(column for block in self.blocks for column in block.required_columns)

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
