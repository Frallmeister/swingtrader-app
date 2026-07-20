# Development Workflow

## Project Layout

Application code lives under `src/swingtrader` and tests live under `tests`.

```text
src/swingtrader/
  core/       Shared configuration, database engine utilities, logging, and contracts.
  data/       Clients, ingestion workflows, bronze storage, features, and jobs.
  modeling/   Future dataset, training, inference, and model registry code.
  web/        Future dashboard and user-facing application code.
tests/        Unit and integration-style tests using local fixtures.
docs/         Project documentation and planning material.
```

## Quality Checks

Format check:

This checks whether source and test files already match the repository's Ruff formatting rules without rewriting files.

```powershell
uv run ruff format --check src tests
```

Lint:

This checks for code issues selected in `pyproject.toml`, including import ordering, common Python mistakes, and simplifications.

```powershell
uv run ruff check src tests
```

Tests:

```powershell
uv run pytest
```

Documentation build:

This builds the static site into `site/`. With `--strict`, the command also acts as a validation check because warnings and documentation errors fail the build.

```powershell
uv run --group docs mkdocs build --strict
```

The generated `site/` directory is ignored by Git and should not be committed.

Serve the documentation locally:

```powershell
uv run --group docs mkdocs serve
```

## Dependency Changes

When dependencies change, update `uv.lock` by running:

```powershell
uv lock
```

This rewrites the lock file so it matches the dependencies currently declared in `pyproject.toml`. Commit the updated `uv.lock` with the dependency change.

## Code Placement

- Provider-specific API behavior belongs in `data.clients`.
- Retrieval orchestration belongs in `data.ingestion`.
- Source-oriented database schemas and writes belong in `data.bronze`.
- Ready-to-use data database initialization belongs in `data.db`; generic engine creation belongs in `core.db`.
- Reusable technical indicators belong in `indicators`; the canonical market-frame contract helpers belong in `data.market_frame`.
- Model-ready feature transformations belong in `data.features`, with the default pipeline in `data.features.pipeline`.
- Thin runnable entrypoints will belong in `data.jobs`.
- Shared, domain-neutral numerical helpers belong in `core.numerical`; other shared infrastructure belongs in `core`.

Keep implementation changes narrow. Avoid using notebooks as the only home for reusable behavior.