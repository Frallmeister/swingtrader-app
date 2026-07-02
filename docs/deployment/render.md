# Render Deployment

Render deployment is planned, not implemented yet.

## Intended Shape

The deployed application is expected to use:

- a database service, likely PostgreSQL;
- a scheduled job for daily market data updates;
- a web service for the dashboard;
- environment variables for configuration and secrets.

## Planned Environment Variables

- `SWINGTRADER_DATABASE_URL`
- future provider credentials if needed
- future application/runtime settings

## Deployment Principles

- No secrets committed to the repository.
- Jobs should be idempotent.
- Logs should contain enough request and row-count information to debug failures.
- The web app should read persisted results rather than triggering heavy data workflows inline.