from fastapi import APIRouter, Request

from app.schemas.alerts import TradingViewAlert

router = APIRouter()


@router.post("/webhook/tradingview")
def tradingview_webhook(alert: TradingViewAlert, request: Request) -> dict:
    services = request.app.state.services
    return services.signal_service.process_alert(alert)
