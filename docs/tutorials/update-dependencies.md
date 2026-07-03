# Update Dependencies

Use this when adding, removing, or changing project dependencies.

## Edit Dependencies

Change the dependency declaration in `pyproject.toml`.

Examples:

- runtime dependencies belong in `[project].dependencies`;
- optional application extras belong in `[project.optional-dependencies]`;
- development-only groups belong in `[dependency-groups]`.

## Update The Lock File

Run:

```powershell
uv lock
```

This updates `uv.lock` so future installs use the same resolved package versions.

## Install The Updated Environment

For full local development, run:

```powershell
uv sync --all-extras --dev --group notebook --group docs
```

## Commit Both Files

Commit `pyproject.toml` and `uv.lock` together.