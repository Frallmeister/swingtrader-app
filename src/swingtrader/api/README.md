# API

This package is reserved for the planned FastAPI backend.

The API layer will own HTTP transport concerns, request and response schemas,
authentication and authorization integration, and calls into application services.
It must not own feature algorithms, model training, scheduled market workflows, or
frontend presentation logic.

Full-market feature generation and model inference should run in scheduled jobs. The
API should read persisted market, prediction, and application state and return bounded
responses to the frontend. Limited chart-data and indicator requests may be calculated
on demand when their cost is predictable and suitably bounded.
