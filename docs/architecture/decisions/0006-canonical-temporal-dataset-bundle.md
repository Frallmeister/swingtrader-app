# ADR 0006: Canonical Temporal Dataset Bundle

- **Status:** Accepted
- **Date:** 2026-07-25

## Context

Feature and target contracts already produce canonical market-indexed frames, but model development needs one stable product that aligns those calculations, records target resolution dates, and preserves enough metadata for leakage-safe temporal splitting. Building model-specific matrices directly from raw prices would duplicate alignment rules and risk truncating expanding or path-dependent feature history at split boundaries.

## Decision

Introduce an unsplit, framework-neutral `TemporalDatasetBundle` containing aligned feature, target, and sample-metadata DataFrames plus a deterministic manifest.

Dataset identity is defined by a lower-level `TemporalDatasetSpec`: feature set, target set, selected supervised task, resolved provider/ticker universe, and inclusive data cutoff. Experiment splits, models, seeds, and MLflow remain downstream.

Features and targets are calculated independently over the full available historical prefix through the cutoff. All output frames share the canonical unique, sorted `MultiIndex` `(provider, ticker, trading_date)`. Rows with an unavailable selected target are excluded from every frame, while missing feature values are retained. Each retained sample records `target_end_date` and cutoff-aware training-eligibility metadata.

Fixed-horizon tasks derive their end date from observed future sessions. Event-based tasks may select an explicit target-set end-date output. Ticker eligibility is metadata, not an implicit universe filter.

## Consequences

- Temporal splitting can purge rows by actual target resolution date without recomputing features or targets.
- Expanding and path-dependent features are reproducible because construction does not begin at a split boundary.
- scikit-learn, XGBoost, PyTorch, and MLflow remain adapters or consumers rather than canonical dataset dependencies.
- Dataset manifests identify construction semantics and diagnostics but do not claim that mutable source data has been snapshotted or content-addressed.
- Materialized dataset persistence, split assignment, preprocessing, and model training require later decisions and implementations.
