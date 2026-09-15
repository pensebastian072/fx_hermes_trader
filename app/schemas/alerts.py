"""TradingView-style alert payload schema."""

import hashlib
from datetime import datetime

from pydantic import BaseModel, field_validator


class TradingViewAlert(BaseModel):
    source: str
    symbol: str
    timeframe: str
    price: float
    strategy: str
    signal: str
    timestamp: datetime
    alert_id: str | None = None

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        # Accept "OANDA:EURUSD", "EUR/USD", "eurusd" -> "EURUSD"
        v = v.split(":")[-1].replace("/", "").strip().upper()
        if not v.isalpha() or len(v) < 6:
            raise ValueError("symbol must be a currency pair like EURUSD")
        return v

    @field_validator("price")
    @classmethod
    def price_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price must be positive")
        return v

    @field_validator("source", "timeframe", "strategy", "signal")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("field must not be empty")
        return v

    def dedupe_key(self) -> str:
        """alert_id wins; otherwise a stable hash of the identifying fields."""
        if self.alert_id:
            return self.alert_id
        raw = "|".join(
            [self.symbol, self.timeframe, f"{self.price}", self.signal, self.timestamp.isoformat()]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
