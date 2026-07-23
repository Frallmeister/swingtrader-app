# ADR 0005: Versioned Feature-Set Contract

- **Status:** Accepted

## Context

The first OHLCV feature catalogue grew through seven independently callable family orchestrators. The default pipeline called those functions with defaults distributed across their modules. That was convenient during feature exploration, but it did not provide a stable identity for the resulting model schema.

A later default-value edit could otherwise change training data without changing the pipeline call. Model and dataset artifacts also need an exact feature column order and parameter record rather than a description such as "all default features".

The repository does not yet know which candidate features will survive model selection, and it does not need a feature store, generic dependency graph, or fine-grained production optimizer before that evidence exists.

## Decision

Feature sets are explicit, immutable, and versioned specifications.

Each feature block records:

- a stable block name;
- the executable family builder;
- explicit parameter values;
- ordered output columns;
- required source columns;
- a bounded, expanding, or path-dependent history classification.

A feature set records an ordered collection of blocks plus a name and version. Different block selections or parameterizations require a different feature-set identity. Its manifest is deterministic and JSON serializable so later dataset and model manifests can embed it.

The initial `ohlcv_v1_candidates:1` contract reproduces the feature output that existed when this ADR was accepted. `add_default_features` remains available as a compatibility wrapper, but it delegates to this specification instead of relying directly on distributed function defaults.

Selection and execution operate at existing feature-family granularity. Individual family builders remain public and independently callable.

## Consequences

- Training and inference can refer to a concrete feature-set identifier and manifest.
- Default parameter changes no longer silently redefine the accepted feature set; a changed schema or parameterization requires a new version.
- Tests can compare declared output columns with actual pipeline output.
- Callers can execute selected families without calculating every family.
- The contract duplicates some output-column metadata already implied by implementation code. Contract tests must detect drift between the declaration and actual output.
- This decision does not yet create immutable data snapshots, model manifests, feature persistence, dynamic dependency resolution, or individual-column execution inside broad families. Those belong to later modeling and production work.
