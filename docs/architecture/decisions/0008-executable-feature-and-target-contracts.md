# ADR 0008: Executable Feature and Target Contracts

- **Status:** Accepted

## Context

ADR 0005 introduced immutable, versioned feature-set declarations so dataset and model artifacts could identify a concrete candidate schema. Target families later adopted a similar manifest-oriented structure.

The declarations still depended on separate pipeline code to validate required inputs, execute builders, detect output collisions, and check declared outputs. A builder could return undeclared columns, omit a declared column, use an unrecorded default, or change the index while the specification and manifest remained unchanged. Feature and target execution also enforced similar rules in different downstream modules.

## Decision

Feature and target specifications are executable contracts rather than descriptive metadata.

- `FeatureBlockSpec` and `TargetFamilySpec` require every keyword-compatible builder parameter to be explicit, pass those parameters directly, validate required inputs and output collisions, require an unchanged index, and return only declared outputs in declaration order.
- `FeatureSetSpec` and `TargetSetSpec` execute their blocks or families in declaration order and append only the enforced outputs to an independent copy of the source frame.
- Later blocks or families may consume outputs declared by earlier ones. External source requirements remain derivable from that ordered contract.
- Compatibility helpers such as `add_feature_set()` delegate to the specifications instead of reimplementing contract validation.
- Temporal dataset construction consumes the executable feature and target sets directly, then performs only dataset-level alignment, target-resolution, eligibility, and manifest work.

Individual feature and target builders remain public and independently callable. This decision does not introduce automatic discovery, a generic dependency graph, individual-column execution within a family, persistence, or a feature store.

## Consequences

- The parameter mapping and output schema in a manifest are the values and columns enforced during execution.
- Undeclared builder outputs cannot leak into a feature or target set.
- Missing inputs, output collisions, missing declared outputs, duplicate builder columns, invalid return types, and index changes fail at the specification boundary.
- Feature and target orchestration become symmetric, and downstream dataset code no longer duplicates their validation logic.
- Builder signature changes can invalidate a specification at construction time, making contract drift visible earlier.
- Existing feature-set or target-set versions do not need a version bump when this refactor preserves their recorded parameters, output order, and numerical results. A future schema or parameter change still requires a new version under ADR 0005.
