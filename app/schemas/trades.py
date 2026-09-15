"""Paper order and fill schemas. Paper only - there is no live order schema."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PaperOrder(BaseModel):
    order_id: str
    timestamp: datetime
    symbol: str
    direction: Literal["long", "short"]
    size: float
    entry: float
    stop: float
    target: float
    risk_percent: float
    status: Literal["open", "closed"] = "open"
    mode: Literal["paper"] = "paper"


class Fill(BaseModel):
    fill_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    direction: Literal["long", "short"]
    kind: Literal["open", "close"]
    price: float
    size: float
    realized_pnl: float = 0.0
