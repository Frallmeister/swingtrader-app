# API

The FastAPI backend exposes the discretionary replay application to the React frontend. It owns HTTP validation and delegates replay rules, indicator calculation, persistence, and portfolio accounting to `swingtrader.replay` services.

Run it from the repository root with:

```bash
uv run --extra api uvicorn swingtrader.api.app:app --reload
```

The interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
