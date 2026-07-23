# ADR 0001: FastAPI and React Application Boundary

- **Status:** Accepted
- **Date:** 2026-07-23

## Context

The original project direction considered a Python dashboard, and the repository contained an empty `swingtrader.web` placeholder describing Dash layouts and callbacks. The current product direction is a TypeScript and React frontend backed by a Python API.

Keeping presentation, HTTP transport, database access, and numerical feature code in one Python web package would blur ownership and make both frontend and backend evolution harder.

## Decision

Use two explicit application boundaries:

- a FastAPI backend under `src/swingtrader/api/`;
- a separate TypeScript and React application under `frontend/`.

The backend owns HTTP transport, explicit request and response schemas, authentication and authorization integration, and calls into application services. It does not own feature algorithms, model training, scheduled workflows, or React presentation code.

The frontend owns presentation, routing, and browser-side state. It communicates with the backend through HTTP and does not access the database or Python package internals. TypeScript API types should be generated from the backend OpenAPI specification where practical.

## Consequences

- Dash is not part of the planned runtime architecture.
- Notebook visualization dependencies such as Plotly may remain for research and EDA.
- Frontend and backend can be deployed and scaled independently.
- API schemas become an explicit compatibility boundary.
- Authentication can be implemented at the API boundary without coupling it to numerical packages.
