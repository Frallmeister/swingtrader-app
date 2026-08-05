from datetime import date

from fastapi.testclient import TestClient

from swingtrader.api.app import app
from swingtrader.api.dependencies import get_replay_service
from swingtrader.replay.service import ReplayService


def test_health_endpoint():
    assert TestClient(app).get("/api/health").json() == {"status": "ok"}


def test_create_and_load_replay_api(seeded_engine):
    service = ReplayService(engine=seeded_engine)
    app.dependency_overrides[get_replay_service] = lambda: service
    try:
        client = TestClient(app)
        response = client.post(
            "/api/replays",
            json={
                "name": "API replay",
                "provider": "yfinance",
                "tickers": ["AAA.ST"],
                "start_date": date(2020, 1, 2).isoformat(),
                "end_date": date(2020, 1, 9).isoformat(),
                "initial_cash": 50_000,
                "courtage_profile": "mini",
                "barrier_policy": "candle_path",
            },
        )
        assert response.status_code == 201
        replay_id = response.json()["session"]["id"]
        assert client.get(f"/api/replays/{replay_id}").status_code == 200
        catalogue = client.get("/api/indicators").json()
        assert any(item["id"] == "macd" for item in catalogue)
    finally:
        app.dependency_overrides.clear()
