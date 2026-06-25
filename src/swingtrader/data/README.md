# Data

Data acquisition, storage, and feature preparation. This package owns external market data clients, reproducible raw data ingestion, database writes for bronze/source data, feature engineering pipelines, and dataset construction logic used by downstream modeling jobs.

## Purpose

The data package turns external market and macroeconomic data into reproducible database records that can be used by modeling and web code. It should own the full path from API retrieval to bronze storage and engineered feature tables.

The first development focus is API exploration and ingestion for:

- `yfinance` for Swedish market data such as OHLC, adjusted prices, volume, dividends, splits, and fund NAV data when available.
- Sveriges Riksbank, FRED, ECB, Statistiska centralbyran, and Konjunkturinstitutet for macroeconomic and financial time series.

## Package Layout

```text
data/
  clients/    Provider-specific API clients and HTTP wrappers.
  ingestion/  Reproducible retrieval and normalization workflows.
  bronze/     Source-oriented database schemas, repositories, and writers.
  features/   Feature engineering from bronze/source data into model inputs.
  jobs/       Thin runnable entrypoints for backfills, daily updates, and rebuilds.
```

## Boundaries

Provider-specific API details belong in `clients`. Code here can know about endpoints, request parameters, response formats, rate limits, retries, and provider quirks.

Retrieval decisions belong in `ingestion`. Code here should decide what symbols or series to fetch, which date windows to request, how to normalize provider responses, and which request metadata is needed for reproducibility.

Source persistence belongs in `bronze`. Bronze tables should stay close to the provider data while adding project-level metadata such as provider name, source identifier, request id, fetch timestamp, and response provenance.

Feature transformations belong in `features`. This code should produce cleaned, aligned, model-ready values from bronze/source records, including market indicators, macro joins, lags, rolling windows, labels, and split-safe feature tables.

Operational commands belong in `jobs`. Jobs should be thin orchestration layers that call reusable client, ingestion, bronze, and feature code. Render cron jobs should eventually run modules from this folder.

## Universe and Ticker Onboarding

Universe configuration defines desired ticker membership. Available universe files, such as Swedish Large Cap, describe which tickers can be considered. The active ticker configuration describes which tickers the model is allowed to train on or trade.

Activating a ticker should be treated as a data onboarding event. Adding a ticker to the active configuration means the ticker is allowed, but it does not by itself mean that the ticker is ready for inference or eligible for training. The data layer should eventually provide workflows that compare the desired active universe with the data already present in the database and then backfill any missing tickers.

The intended onboarding workflow is:

1. Validate that the ticker exists in an available universe.
2. Download historical market data for the ticker.
3. Upsert the historical data into the bronze layer.
4. Build or rebuild engineered features for the ticker.
5. Validate that required data and features are complete enough for inference and future training.

This makes the active ticker configuration the desired state and the database the actual state. Daily update jobs can then focus on already-onboarded active tickers, while a separate sync or activation job can handle newly active tickers that require historical backfill.

## Bronze Data Principles

Bronze data should answer four questions:

- Where did this value come from?
- What API request produced it?
- When was it fetched?
- Can features be rebuilt from it without downloading the same data again?

The bronze layer should not contain model-ready feature joins, train/validation/test splits, imputed values, or prediction outputs. Those belong in feature and modeling layers.

For API responses, prefer storing a lightly normalized bronze table as the main query surface and keeping raw response metadata for audit and replay. Small raw responses can be stored directly in the database; larger responses can later move to file or object storage with a content hash stored in the database.

## Provider Exploration Workflow

Notebooks are useful for learning each provider API, but reusable logic should move into `src/swingtrader/data` once it stabilizes.

Recommended flow:

1. Explore a provider in `notebooks/data_exploration`.
2. Record the provider identifiers, request parameters, response shape, units, frequency, and date semantics.
3. Move stable request code into `clients`.
4. Move normalization and reproducible retrieval logic into `ingestion`.
5. Write source-oriented records through `bronze`.
6. Build model-facing transformations in `features`.

## Time Series Notes

Market data and macro data need different care.

Market data is usually keyed by symbol and trading date. The first scope is Swedish Large Cap securities through `yfinance`, with enough metadata to distinguish prices, adjusted prices, corporate actions, currency, exchange, and fetch time.

Macro data is usually keyed by provider series id and observation date. Many macro series also need release dates, revisions, units, seasonal adjustment metadata, and frequency. Avoid assuming that an observation was known on its observation date.

Point-in-time correctness matters for modeling. Feature code should avoid using macro values before they would have been available to a real trading decision.

## Dependency Direction

The intended dependency direction is:

```text
jobs -> ingestion -> clients
jobs -> bronze
jobs -> features -> bronze
```

The `data` package may import shared utilities from `swingtrader.core`. It should not import implementation code from `swingtrader.modeling` or `swingtrader.web`.
