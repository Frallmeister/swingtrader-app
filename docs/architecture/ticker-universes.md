# Ticker Universes

Ticker universe configuration describes which symbols the project may consider.

## Available Universes

Available universe files are curated YAML artifacts under `src/swingtrader/configs/universes`. They describe ticker catalogs such as Swedish Large Cap and Mid Cap.

These files are source-controlled and reviewed. They are not generated dynamically at runtime.

## Local Universe Generation

The helper module `swingtrader.data.ingestion.universe` and the notebook `notebooks/workflows/data/00_create_ticker_universes.ipynb` are local/bootstrap tools for creating or refreshing available universe YAML files.

Generation requires an explicit output path and does not overwrite existing files unless requested. This keeps runtime configuration separate from local provider metadata fetching.

## Active Trading Universe

The active trading universe is the production candidate set: tickers the deployed app should keep updated and eventually rank as trade candidates.

Runtime code reads `active_tickers.yml` and the referenced available universe files through `resolve_active_tickers()`.

## Future Training Universe

The training universe may be broader than the active trading universe. It may include inactive tickers, all curated Swedish universes, and eventually non-Swedish markets such as US equities.

This is planned work. Modeling code should not assume that active tickers are always the only training candidates.

## Desired And Actual State

- Desired state: ticker configuration.
- Actual state: data currently available in bronze and future feature tables.

Activating a ticker means it is allowed. It does not automatically mean the ticker is ready for inference or training.