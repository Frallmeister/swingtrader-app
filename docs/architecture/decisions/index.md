# Architecture Decisions

Architecture Decision Records (ADRs) document choices that affect multiple packages, workflows, or future implementations. They capture why a direction was chosen, not only what the current code happens to do.

## Status Values

- **Proposed**: under review and not yet a project contract.
- **Accepted**: the project should follow the decision.
- **Superseded**: replaced by a later ADR.
- **Rejected**: considered and deliberately not adopted.

Accepted ADRs should not be silently rewritten when the architecture changes. Add a new ADR that supersedes the previous decision and link the two records.

## Accepted Decisions

- [ADR 0001: FastAPI and React application boundary](0001-fastapi-react-application-boundary.md)
- [ADR 0002: Scheduled inference and persisted predictions](0002-scheduled-inference-and-persisted-predictions.md)
- [ADR 0003: Separate research targets from executable evaluation](0003-separate-research-targets-from-executable-evaluation.md)
- [ADR 0004: Adjustment-consistent model prices](0004-adjustment-consistent-model-prices.md)
- [ADR 0005: Versioned feature-set contract](0005-versioned-feature-set-contract.md)
- [ADR 0006: Canonical temporal dataset bundle](0006-canonical-temporal-dataset-bundle.md)
- [ADR 0007: Purged fixed temporal splitting](0007-purged-fixed-temporal-splitting.md)
- [ADR 0008: Executable feature and target contracts](0008-executable-feature-and-target-contracts.md)
