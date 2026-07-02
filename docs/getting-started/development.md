# Development Workflow

## Project Layout

Application code lives under `src/swingtrader` and tests live under `tests`.

```text
src/swingtrader/
  core/       Shared configuration, database, logging, schemas, and contracts.
  data/       Clients, ingestion workflows, bronze storage, features, and jobs.
  modeling/   Future dataset, training, inference, and model registry code.
  web/        Future dashboard and user-facing application code.
tests/        Unit and integration-style tests using local fixtures.
docs/         Project documentation and planning material.
```

## Quality Checks

Format check:

```powershell
uv run ruff format --check src tests
```

Lint:

```powershell
uv run ruff check src tests
```

Tests:

```powershell
uv run pytest
```

Documentation build:

```powershell
uv run mkdocs build --strict
```

## Dependency Changes

When dependencies change, update `uv.lock` and commit it with the code change:

```powershell
uv lock
```

## Code Placement

- Provider-specific API behavior belongs in `data.clients`.
- Retrieval orchestration belongs in `data.ingestion`.
- Source-oriented database schemas and writes belong in `data.bronze`.
- Model-ready transformations will belong in `data.features`.
- Thin runnable entrypoints will belong in `data.jobs`.
- Shared infrastructure belongs in `core`.

Keep implementation changes narrow. Avoid using notebooks as the only home for reusable behavior.