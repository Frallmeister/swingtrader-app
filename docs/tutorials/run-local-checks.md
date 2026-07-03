# Run Local Checks

Use this before opening a pull request or after changing application code.

## Format Check

```powershell
uv run ruff format --check src tests
```

If this fails, format the code:

```powershell
uv run ruff format src tests
```

## Lint

```powershell
uv run ruff check src tests
```

## Tests

```powershell
uv run pytest
```

## Documentation

```powershell
uv run --group docs mkdocs build --strict
```

The documentation build writes generated files to `site/`. That directory is ignored by Git.