# Troubleshooting

## `uv` Hardlink Warnings On Windows

Set `UV_LINK_MODE=copy` when the project and cache are on different filesystems.

```powershell
$env:UV_LINK_MODE = "copy"
```

Persist it for your Windows user account:

```powershell
[Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")
```

## SQLite Datetime Round Trips

SQLite may return naive datetimes even when SQLAlchemy columns use `DateTime(timezone=True)`. Tests should not expect timezone-aware round trips from SQLite.

## Notebook Outputs

Install nbstripout once per clone:

```powershell
uv run nbstripout --install
```

Avoid committing generated notebook output unless there is a specific reason.

## Common Validation Commands

```powershell
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
uv run mkdocs build --strict
```