# Swingtrader

## Project Direction

Swingtrader is being built around a data-first workflow for market and macroeconomic data. During development, the project will download historical data for configured active tickers, store source-oriented records in a bronze layer, build engineered features, and train models from reproducible train, validation, and test splits.

After deployment, the intended workflow is to run daily data updates that append new quotes and market data to bronze storage, refresh feature tables, and run the current production model on the latest feature data. New model versions can then be trained periodically as more data becomes available.

Ticker activation should be treated as a data onboarding workflow, not just a config edit. The active universe defines which tickers are allowed, while bronze backfills, feature generation, and validation determine whether a ticker is ready for inference and eligible for future training runs.

## Development

Use `uv` for dependency and environment management:

```powershell
uv sync --all-extras --dev --group notebook
```

This installs all application extras, development tools, and notebook tooling for local development.

The notebook tooling includes JupyterLab, ipykernel, ipywidgets, tqdm, nbstripout, and Jupytext. Install the nbstripout Git filter once per clone to avoid committing generated notebook outputs:

```powershell
uv run nbstripout --install
```

Jupytext is included for pairing notebooks with plain text representations when useful for review or version control.

If `uv` reports a hardlink fallback warning on Windows, set `UV_LINK_MODE` to `copy` for the current PowerShell session before running dependency commands:

```powershell
$env:UV_LINK_MODE = "copy"
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

## Historical Bronze Ingestion

Historical daily market data ingestion is available as a library function. It resolves active
tickers when no explicit ticker list is provided, downloads yfinance daily prices, and upserts
the resulting rows into `bronze_market_daily_prices`.

By default, local ingestion uses SQLite at `data/swingtrader.sqlite`. Set
`SWINGTRADER_DATABASE_URL` to override this, for example when using PostgreSQL or a different
SQLite file.

```python
from datetime import date

from swingtrader.data.ingestion.market_data import ingest_historical_daily_prices

result = ingest_historical_daily_prices(
	start_date=date(2024, 1, 1),
	end_date=date(2024, 2, 1),
	limit=3,
)

print(result)
```

This is intentionally not exposed as a command yet. The runnable daily update job is tracked
separately and will wrap this library surface later.

## Copyright

Copyright © 2026 [Fredrik Hansson]. All rights reserved.

This software and its source code are proprietary and confidential.
No permission is granted to use, copy, modify, distribute, sublicense,
or create derivative works from this software, in whole or in part,
without prior written permission from the copyright holder.
