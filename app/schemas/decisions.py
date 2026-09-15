"""Decision trace schema - every alert produces exactly one trace."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DecisionTrace(BaseModel):
    timestamp: datetime
    symbol: str
    strategy: str
    signal: str
    direction: Literal["long", "short", "flat"]
    final_score: int
    decision: Literal["accept", "reject"]
    reason: str
    rejection_reason: str | None = None
    risk_checks_failed: list[str] = Field(default_factory=list)
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    size: float | None = None
    risk_percent: float | None = None
    order_id: str | None = None
    mode: str = "paper"
