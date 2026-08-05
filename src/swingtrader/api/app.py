"""FastAPI application for the local discretionary replay frontend."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from swingtrader.api.dependencies import get_replay_service
from swingtrader.api.schemas import (
    ChartRequest,
    CreatePresetRequest,
    CreateReplayRequest,
    EveningDecisionRequest,
    MorningRevisionRequest,
    ScreeningConfiguration,
    UpdatePresetRequest,
    WatchlistRequest,
)
from swingtrader.replay.service import (
    InsufficientCashError,
    ReplayService,
    ReplayValidationError,
)

app = FastAPI(title="SwingTrader Replay API", version="0.1.0")
ReplayServiceDependency = Annotated[ReplayService, Depends(get_replay_service)]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ReplayValidationError)
def replay_validation_error(_request: Any, error: ReplayValidationError) -> Any:
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(error)})


@app.exception_handler(KeyError)
def unknown_replay_error(_request: Any, error: KeyError) -> Any:
    return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(error)})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/indicators")
def indicators(service: ReplayServiceDependency) -> list[dict[str, Any]]:
    return service.indicator_catalogue()


@app.post("/api/replays", status_code=status.HTTP_201_CREATED)
def create_replay(
    request: CreateReplayRequest,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.create_replay(**request.model_dump())


@app.get("/api/replays")
def list_replays(service: ReplayServiceDependency) -> list[dict[str, Any]]:
    return service.list_replays()


@app.get("/api/replays/{replay_id}")
def get_replay(
    replay_id: str, service: ReplayServiceDependency
) -> dict[str, Any]:
    return service.get_state(replay_id)


@app.post("/api/replays/{replay_id}/decisions")
def save_decision(
    replay_id: str,
    request: EveningDecisionRequest,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.save_evening_decision(replay_id, **request.model_dump())


@app.post("/api/replays/{replay_id}/finalize-evening")
def finalize_evening(
    replay_id: str, service: ReplayServiceDependency
) -> dict[str, Any]:
    return service.finalize_evening(replay_id)


@app.patch("/api/replays/{replay_id}/decisions/{decision_id}")
def revise_morning_decision(
    replay_id: str,
    decision_id: str,
    request: MorningRevisionRequest,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.revise_morning_decision(
        replay_id, decision_id, cancelled=request.cancelled, **request.changes()
    )


@app.post("/api/replays/{replay_id}/complete-morning")
def complete_morning(
    replay_id: str, service: ReplayServiceDependency
) -> dict[str, Any]:
    try:
        return service.complete_morning(replay_id)
    except InsufficientCashError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error


@app.post("/api/replays/{replay_id}/chart/{ticker}")
def chart(
    replay_id: str,
    ticker: str,
    request: ChartRequest,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.chart(
        replay_id,
        ticker,
        [indicator.model_dump() for indicator in request.indicators],
        lookback_sessions=request.lookback_sessions,
    )


@app.post("/api/replays/{replay_id}/screen")
def run_screen(
    replay_id: str,
    request: ScreeningConfiguration,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.run_screen(replay_id, request.model_dump())


@app.post("/api/replays/{replay_id}/watchlist")
def add_watchlist(
    replay_id: str,
    request: WatchlistRequest,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.add_watchlist(replay_id, request.ticker, request.note)


@app.delete("/api/replays/{replay_id}/watchlist/{item_id}")
def remove_watchlist(
    replay_id: str,
    item_id: str,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.remove_watchlist(replay_id, item_id)


@app.get("/api/screening-presets")
def list_presets(service: ReplayServiceDependency) -> list[dict[str, Any]]:
    return service.list_presets()


@app.post("/api/screening-presets", status_code=status.HTTP_201_CREATED)
def create_preset(
    request: CreatePresetRequest,
    service: ReplayServiceDependency,
) -> dict[str, Any]:
    return service.create_preset(
        name=request.name,
        description=request.description,
        configuration=request.configuration.model_dump(),
    )


@app.patch("/api/screening-presets/{preset_id}")
def update_preset(
    preset_id: str,
    request: UpdatePresetRequest,
    service: ReplayServiceDependency,
) -> list[dict[str, Any]]:
    values = request.model_dump(exclude_none=True)
    if request.configuration is not None:
        values["configuration"] = request.configuration.model_dump()
    return service.update_preset(preset_id, **values)


@app.delete("/api/screening-presets/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(
    preset_id: str, service: ReplayServiceDependency
) -> Response:
    service.delete_preset(preset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
