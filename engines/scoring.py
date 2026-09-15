"""Milestone 1 scoring stub.

Deterministic mapping from TradingView signal names to scores, plus synthetic
stop/target geometry. Real engines (trend, mean reversion, chart patterns,
rate differentials, regime) replace this in later milestones - see
PROJECT_PLAN.md sections 6 and 8.
"""

from app.schemas.alerts import TradingViewAlert
from app.schemas.signals import ScoreResult

DEFAULT_SIGNAL_SCORES = {
    "long_candidate": 75,
    "short_candidate": -75,
    "long_strong": 85,
    "short_strong": -85,
    "watch": 0,
}


class ScoringStub:
    def __init__(self, scoring_config: dict | None = None):
        cfg = scoring_config or {}
        self.signal_scores: dict[str, int] = cfg.get("signal_scores", DEFAULT_SIGNAL_SCORES)
        self.long_threshold = int(cfg.get("long_threshold", 70))
        self.short_threshold = int(cfg.get("short_threshold", -70))
        self.stop_distance_pct = float(cfg.get("stop_distance_pct", 0.003))
        self.reward_risk = float(cfg.get("reward_risk", 2.0))

    def score(self, alert: TradingViewAlert) -> ScoreResult:
        raw = int(self.signal_scores.get(alert.signal, 0))
        if raw >= self.long_threshold:
            direction = "long"
        elif raw <= self.short_threshold:
            direction = "short"
        else:
            direction = "flat"

        entry = stop = target = None
        if direction != "flat":
            entry = alert.price
            stop_distance = alert.price * self.stop_distance_pct
            sign = 1 if direction == "long" else -1
            stop = entry - sign * stop_distance
            target = entry + sign * self.reward_risk * stop_distance

        reason = (
            f"Stub scoring: signal '{alert.signal}' maps to {raw} "
            f"(thresholds {self.short_threshold}/{self.long_threshold})."
        )
        return ScoreResult(
            symbol=alert.symbol,
            direction=direction,
            final_score=raw,
            reason=reason,
            entry=entry,
            stop=stop,
            target=target,
        )
