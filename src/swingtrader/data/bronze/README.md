tables derived from the packaged universe files.
each ingestion attempt may have a new request id while describing the same market observation.
the bronze table reproducible for local SQLite while mapping cleanly to Postgres `ON CONFLICT`
the provider fetch time, not merely the database insert time. Ingestion code should write UTC
timestamps.
development and moved to Postgres with minimal changes when the app is deployed. Price and
# Bronze

Bronze storage keeps source-oriented market data with enough metadata to rerun ingestion safely and rebuild downstream features.

See [Bronze storage](../../../../docs/data/bronze.md) for table details, idempotency behavior, and portability notes.
