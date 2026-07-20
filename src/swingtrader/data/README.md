# Data

Data acquisition, storage, and feature preparation live here. The package owns provider clients, ingestion workflows, source-oriented bronze storage, ticker eligibility checks, and reusable feature transformations used by modeling and web code.

See the main documentation for details:

- [Data layer overview](../../../docs/data/overview.md)
- [Ingestion](../../../docs/data/ingestion.md)
- [Bronze storage](../../../docs/data/bronze.md)
- [Ticker onboarding](../../../docs/data/ticker-onboarding.md)
- [Ticker eligibility](../../../docs/data/eligibility.md)
- [Ticker universes](../../../docs/architecture/ticker-universes.md)
- [Features](../../../docs/data/features.md)

The intended local package boundaries remain:

```text
clients/        Provider-specific API clients and HTTP wrappers.
ingestion/      Retrieval and normalization workflows.
bronze/         Source-oriented schemas and writers.
market_frame.py Canonical market-frame index contract and per-ticker helpers.
features/       Model-ready in-memory transformations.
jobs/           Thin runnable entrypoints for local data workflows.
```
