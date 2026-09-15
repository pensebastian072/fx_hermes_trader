"""Scoring output schema."""

from typing import Literal

from pydantic import BaseModel


class ScoreResult(BaseModel):
    symbol: str
    direction: Literal["long", "short", "flat"]
    final_score: int
    reason: str
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
