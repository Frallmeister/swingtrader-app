# Replay frontend

This directory contains the TypeScript and React user interface for discretionary historical replay. It calls the FastAPI backend under `/api`; it never reads SQLite or imports Python code directly.

The first version includes:

- explicit evening and morning modes;
- mandatory decisions for every open position;
- new entries, full exits, and partial reductions;
- configurable indicators with grouped multi-output visibility;
- continuous indicator screening and reusable screening presets;
- watchlists, portfolio metrics, metric history, and position-duration analysis;
- TradingView Lightweight Charts using only data released by the replay backend.

Run `npm install` once, then `npm run dev`. See the replay documentation for the full backend and frontend startup sequence.
