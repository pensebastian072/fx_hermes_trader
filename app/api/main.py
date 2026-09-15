"""FastAPI application factory.

Run with: uvicorn app.api.main:app --reload
"""

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import health, signals, webhooks
from app.deps import build_services
from app.paths import logs_dir
from app.services.artifact_service import append_jsonl


def create_app() -> FastAPI:
    app = FastAPI(
        title="FX Hermes Trader",
        description="Paper-trading FX research harness. Live trading is not implemented.",
        version="0.1.0",
    )
    app.state.services = build_services()
    app.include_router(health.router)
    app.include_router(webhooks.router)
    app.include_router(signals.router)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = [
            {"loc": [str(p) for p in e.get("loc", [])], "msg": e.get("msg"), "type": e.get("type")}
            for e in exc.errors()
        ]
        # Invalid webhook payloads are logged, not hidden.
        if request.url.path == "/webhook/tradingview":
            body = await request.body()
            append_jsonl(
                logs_dir() / "rejected_alerts.jsonl",
                {
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "rejection_reason": "schema validation failed",
                    "errors": errors,
                    "raw_body": body.decode("utf-8", errors="replace")[:2000],
                },
            )
        return JSONResponse(status_code=422, content={"status": "rejected", "errors": errors})

    return app


app = create_app()
