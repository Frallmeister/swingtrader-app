# Frontend

This directory is reserved for the planned TypeScript and React frontend.

The frontend will own presentation, browser-side state, routing, and calls to the
FastAPI backend. It must not access the database directly or depend on Python package
internals. Shared request and response types should be generated from the backend's
OpenAPI specification where practical.

The frontend application has not been initialized yet. Add its package manager,
build tooling, and source tree when frontend implementation begins.
