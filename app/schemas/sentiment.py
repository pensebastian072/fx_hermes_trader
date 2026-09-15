"""Central-bank sentiment output schema (PROJECT_PLAN.md 6.5 / 29.3).

Sentiment is bias-only by construction: trade_action is a frozen literal and
cannot represent a direct trade.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AgentStance(BaseModel):
    agent: str
    prior: Literal["dove", "neutral", "hawk"]
    score: float  # -100 (max dovish) .. +100 (max hawkish)


class SentimentConsensus(BaseModel):
    timestamp: datetime
    currency: str
    macro_bias: Literal["bullish", "bearish", "neutral"]
    hawkish_dovish_score: float  # -100 .. +100
    confidence: float  # 0 .. 1
    reason: str
    agent_stances: list[AgentStance] = Field(default_factory=list)
    debate_rounds: int = 0
    inflation_above_target: bool | None = None
    trade_action: Literal["bias_only_no_direct_trade"] = "bias_only_no_direct_trade"
