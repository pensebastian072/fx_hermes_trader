"""Regime snapshot schema - output of the HMM regime detector."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class RegimeSnapshot(BaseModel):
    timestamp: datetime
    symbol: str
    probabilities: dict[str, float]  # regime label -> P(regime | data so far)
    dominant_regime: str
    crisis_probability: float = 0.0
    n_bars: int = 0

    @model_validator(mode="after")
    def check_probabilities(self):
        total = sum(self.probabilities.values())
        if self.probabilities and not 0.99 <= total <= 1.01:
            raise ValueError(f"regime probabilities must sum to ~1, got {total}")
        return self
