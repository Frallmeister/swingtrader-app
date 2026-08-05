# Run the Replay Application

The application has two processes: a Python FastAPI backend and a React development server. Both use the existing `data/swingtrader.sqlite` database unless `SWINGTRADER_DATABASE_URL` is set.

## Prerequisites

Install:

- Python 3.12 or newer;
- `uv`;
- Node.js 22 LTS, which includes `npm`.

The bronze database must already contain daily rows for the tickers and dates used by a replay.

## 1. Install Python dependencies

From the repository root:

```powershell
uv sync --extra data --extra api --dev
```

## 2. Start FastAPI

Keep this terminal open:

```powershell
uv run --extra api uvicorn swingtrader.api.app:app --reload
```

The API runs at `http://127.0.0.1:8000`. Its interactive API documentation is at `http://127.0.0.1:8000/docs`.

## 3. Install frontend dependencies

Open a second terminal:

```powershell
cd frontend
npm install
```

`npm install` creates `frontend/node_modules`. It only needs to be repeated when `package.json` changes materially or the local installation is removed.

## 4. Start React

Still in `frontend`:

```powershell
npm run dev
```

Open the local address printed by Vite, normally `http://localhost:5173`. Vite proxies `/api` requests to FastAPI on port 8000.

## Stop the application

Press `Ctrl+C` in each terminal. Replay state is persisted in SQLite and can be selected again after restarting both processes.

## Checks

Backend tests:

```powershell
uv run pytest tests/replay tests/api
```

Frontend build and tests:

```powershell
cd frontend
npm run build
npm test
```
