# ADR 0002: Scheduled Inference and Persisted Predictions

- **Status:** Accepted
- **Date:** 2026-07-23

## Context

Full-universe feature generation already includes expensive, expanding, and path-dependent calculations. Small production services should not recompute the complete research feature catalogue while a user waits for an HTTP response.

The application is designed for daily or otherwise scheduled decision support rather than intraday order execution. This allows production inference to be separated from interactive reads.

## Decision

Run market updates, selected production feature calculation, and model inference in scheduled jobs. Persist dated prediction and ranking snapshots before the frontend requests them.

The FastAPI backend reads persisted outputs and returns bounded responses. It may calculate limited chart data or indicators on demand only when the cost is predictable and bounded. It must not trigger full-market feature generation or production inference in ordinary request handlers.

Persisted predictions should identify at least the model version, feature-set version, data cutoff, and calculation date.

## Consequences

- API latency and memory usage remain predictable.
- Scheduled jobs can be retried, monitored, and allocated different resources from the API.
- Production feature calculation can be optimized after model selection rather than optimizing every research candidate.
- The frontend may display the age and provenance of the latest prediction snapshot.
- Prediction persistence becomes part of the production architecture even if research features remain in memory.
