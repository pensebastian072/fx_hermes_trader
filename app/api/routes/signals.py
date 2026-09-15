"""Read-only views over the append-only logs."""

from fastapi import APIRouter, Request

from app.paths import logs_dir
from app.services.artifact_service import read_jsonl

router = APIRouter()


@router.get("/signals/recent")
def recent_decisions(limit: int = 20) -> list[dict]:
    return read_jsonl(logs_dir() / "decision_traces.jsonl", limit=limit)


@router.get("/alerts/recent")
def recent_alerts(limit: int = 20) -> list[dict]:
    return read_jsonl(logs_dir() / "alerts.jsonl", limit=limit)


@router.get("/positions")
def open_positions(request: Request) -> list[dict]:
    services = request.app.state.services
    return [o.model_dump(mode="json") for o in services.broker.open_positions()]
