# Swingtrader

Swingtrader is a personal swing-trading decision-support application built around reproducible market data ingestion, bronze storage, feature generation, model training, and eventually ranked trade candidates for manual review.

The project documentation is the main living reference for implemented behavior, planned architecture, and open design decisions:

- [Documentation home](docs/index.md)
- [Installation](docs/getting-started/installation.md)
- [Development workflow](docs/getting-started/development.md)
- [First ingestion](docs/getting-started/first-ingestion.md)
- [Architecture overview](docs/architecture/overview.md)
- [Roadmap](docs/architecture/roadmap.md)

## Quick Start

Use `uv` for dependency and environment management:

```powershell
uv sync --all-extras --dev --group notebook --group docs
```

Run the standard local checks:

```powershell
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
uv run --group docs mkdocs build --strict
```

Serve the documentation locally:

```powershell
uv run --group docs mkdocs serve
```

Keep application code in `src/swingtrader`, tests in `tests`, and long-form guidance in `docs`.

## Copyright

Copyright © 2026 [Fredrik Hansson]. All rights reserved.

This software and its source code are proprietary and confidential.
No permission is granted to use, copy, modify, distribute, sublicense,
or create derivative works from this software, in whole or in part,
without prior written permission from the copyright holder.
