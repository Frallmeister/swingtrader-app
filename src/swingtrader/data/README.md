# Data

Data acquisition, storage, and feature preparation live here. The package owns provider clients, ingestion workflows, source-oriented bronze storage, and planned feature generation used by modeling and web code.

See the main documentation for details:

- [Data layer overview](../../../docs/data/overview.md)
- [Ingestion](../../../docs/data/ingestion.md)
- [Bronze storage](../../../docs/data/bronze.md)
- [Ticker onboarding](../../../docs/data/ticker-onboarding.md)
- [Ticker universes](../../../docs/architecture/ticker-universes.md)
- [Features](../../../docs/data/features.md)

The intended local package boundaries remain:

```text
clients/    Provider-specific API clients and HTTP wrappers.
ingestion/  Retrieval and normalization workflows.
bronze/     Source-oriented schemas and writers.
features/   Planned model-ready transformations.
jobs/       Planned runnable entrypoints.
```
