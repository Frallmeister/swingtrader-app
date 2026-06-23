# swingtrader

Copyright © 2026 [Fredrik Hansson]. All rights reserved.

This software and its source code are proprietary and confidential.
No permission is granted to use, copy, modify, distribute, sublicense,
or create derivative works from this software, in whole or in part,
without prior written permission from the copyright holder.

## Development

Use `uv` for dependency and environment management:

```powershell
uv sync --dev
```

If `uv` reports a hardlink fallback warning on Windows, set `UV_LINK_MODE` to `copy` for the current PowerShell session before running dependency commands:

```powershell
$env:UV_LINK_MODE = "copy"
uv sync --dev
```

To persist this setting for your Windows user account, run:

```powershell
[Environment]::SetEnvironmentVariable("UV_LINK_MODE", "copy", "User")
```

Open a new PowerShell terminal after setting it permanently.

Run code quality checks with Ruff:

```powershell
uv run ruff format .
uv run ruff check .
```

Run tests with pytest:

```powershell
uv run pytest
```

Keep application code in `src/swingtrader` and tests in `tests`. Commit `uv.lock` for reproducible installs.
