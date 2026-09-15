from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    services = request.app.state.services
    return {
        "status": "ok",
        "mode": services.mode,
        "live_trading": False,
        "open_paper_positions": len(services.broker.open_positions()),
        "equity": services.broker.equity,
    }
