"""Shared execution checks for DataFrame-producing specifications.

The helpers stay domain-neutral so feature and target contracts can enforce the
same builder-signature and DataFrame-boundary rules without duplicating code.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from types import MappingProxyType

import pandas as pd

type ContractParameter = bool | int | float | str | tuple[object, ...] | None
type DataFrameBuilder = Callable[..., pd.DataFrame]


def normalize_builder_parameters(
    builder: DataFrameBuilder,
    parameters: Mapping[str, ContractParameter],
    *,
    subject: str,
) -> Mapping[str, ContractParameter]:
    """Validate a builder signature and freeze explicit keyword values."""
    signature = inspect.signature(builder)
    builder_parameters = tuple(signature.parameters.values())
    if not builder_parameters:
        raise ValueError(f"Builder for {subject} must accept a DataFrame.")

    first_parameter = builder_parameters[0]
    if first_parameter.kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise ValueError(
            f"Builder for {subject} must accept its DataFrame as the first positional argument."
        )

    configurable_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }
    configurable_parameters = {
        parameter.name: parameter
        for parameter in builder_parameters[1:]
        if parameter.kind in configurable_kinds
    }
    positional_only = {
        parameter.name
        for parameter in builder_parameters[1:]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    }
    if positional_only:
        names = ", ".join(sorted(positional_only))
        raise ValueError(f"Builder parameters for {subject} must be keyword-compatible: {names}.")

    unknown_parameters = set(parameters).difference(configurable_parameters)
    if unknown_parameters:
        names = ", ".join(sorted(unknown_parameters))
        raise ValueError(f"Unknown parameters for {subject}: {names}.")

    missing_parameters = set(configurable_parameters).difference(parameters)
    if missing_parameters:
        names = ", ".join(sorted(missing_parameters))
        raise ValueError(f"Missing required parameters for {subject}: {names}.")
    return MappingProxyType(dict(parameters))


def execute_dataframe_contract(
    data: pd.DataFrame,
    *,
    builder: DataFrameBuilder,
    parameters: Mapping[str, ContractParameter],
    required_columns: frozenset[str],
    output_columns: tuple[str, ...],
    subject: str,
) -> pd.DataFrame:
    """Execute a builder and return only its validated declared outputs."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{subject} input must be a pandas DataFrame.")
    if not data.columns.is_unique:
        raise ValueError(f"{subject} input contains duplicate columns.")

    missing = sorted(required_columns.difference(_available_input_names(data)))
    if missing:
        raise ValueError(f"{subject} is missing required inputs: {', '.join(missing)}")

    collisions = sorted(set(output_columns).intersection(data.columns))
    if collisions:
        raise ValueError(f"{subject} would overwrite columns: {', '.join(collisions)}")

    original_index = data.index.copy()
    result = builder(data.copy(deep=True), **parameters)
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"{subject} builder must return a pandas DataFrame.")
    try:
        pd.testing.assert_index_equal(original_index, result.index, exact=True)
    except AssertionError as exc:
        raise ValueError(f"{subject} changed the canonical sample index.") from exc
    if not result.columns.is_unique:
        raise ValueError(f"{subject} returned duplicate columns.")

    missing_outputs = sorted(set(output_columns).difference(result.columns))
    if missing_outputs:
        raise ValueError(f"{subject} did not produce columns: {', '.join(missing_outputs)}")
    return result.loc[:, list(output_columns)].copy()


def _available_input_names(data: pd.DataFrame) -> set[str]:
    available = {column for column in data.columns if isinstance(column, str)}
    available.update(name for name in data.index.names if isinstance(name, str))
    return available
