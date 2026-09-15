"""Deterministic risk engine. Final veto power over every trade.

No LLM input reaches this module. All limits come from configs/active/risk.yaml.
"""

from dataclasses import dataclass, field
from datetime import datetime

from risk.event_blackout import is_blackout


@dataclass
class PortfolioState:
    open_trades: int = 0
    trades_today: int = 0
    trades_today_for_pair: int = 0
    daily_pnl_pct: float = 0.0
    weekly_pnl_pct: float = 0.0
    close_only: bool = False
    # P(HIGH_VOL_CRISIS) from the HMM regime detector; 0.0 when no snapshot exists
    crisis_probability: float = 0.0


@dataclass
class RiskDecision:
    allowed: bool
    reasons: list[str] = field(default_factory=list)


class RiskEngine:
    def __init__(self, risk_config: dict, mode: str = "paper"):
        self.mode = mode
        self.starting_capital = float(risk_config.get("starting_capital", 10000))
        self.base_risk_per_trade = float(risk_config.get("base_risk_per_trade", 0.0025))
        self.max_risk_per_trade = float(risk_config.get("max_risk_per_trade", 0.005))
        self.max_daily_loss = float(risk_config.get("max_daily_loss", 0.01))
        self.max_weekly_loss = float(risk_config.get("max_weekly_loss", 0.03))
        self.max_open_trades = int(risk_config.get("max_open_trades", 4))
        self.max_trades_per_day = int(risk_config.get("max_trades_per_day", 6))
        self.max_trades_per_pair_per_day = int(
            risk_config.get("max_trades_per_pair_per_day", 2)
        )
        self.min_reward_risk = float(risk_config.get("min_reward_risk", 1.5))
        self.close_only = bool(risk_config.get("close_only", False))
        self.blackout_calendar = risk_config.get("event_blackout_calendar") or []
        scaling = risk_config.get("regime_scaling") or {}
        self.regime_scaling_enabled = bool(scaling.get("enabled", False))
        self.regime_scale_start = float(scaling.get("scale_start_probability", 0.5))
        self.regime_halt_probability = float(scaling.get("halt_probability", 0.8))

    def evaluate(
        self, symbol: str, timestamp: datetime, state: PortfolioState
    ) -> RiskDecision:
        """Check every limit; collect all failures rather than stopping at the first."""
        reasons: list[str] = []

        if self.mode != "paper":
            reasons.append(
                f"mode '{self.mode}' is not allowed: only paper mode exists in Milestone 1"
            )
        if self.close_only or state.close_only:
            reasons.append("close-only mode active: no new positions")
        if state.daily_pnl_pct <= -self.max_daily_loss:
            reasons.append(
                f"max daily loss reached ({state.daily_pnl_pct:.4f} <= -{self.max_daily_loss})"
            )
        if state.weekly_pnl_pct <= -self.max_weekly_loss:
            reasons.append(
                f"max weekly loss reached ({state.weekly_pnl_pct:.4f} <= -{self.max_weekly_loss})"
            )
        if state.open_trades >= self.max_open_trades:
            reasons.append(f"max open trades reached ({state.open_trades}/{self.max_open_trades})")
        if state.trades_today >= self.max_trades_per_day:
            reasons.append(
                f"max trades per day reached ({state.trades_today}/{self.max_trades_per_day})"
            )
        if state.trades_today_for_pair >= self.max_trades_per_pair_per_day:
            reasons.append(
                f"max trades for {symbol} today reached "
                f"({state.trades_today_for_pair}/{self.max_trades_per_pair_per_day})"
            )
        blackout, event_name = is_blackout(timestamp, self.blackout_calendar)
        if blackout:
            reasons.append(f"event blackout active: {event_name}")
        if (
            self.regime_scaling_enabled
            and state.crisis_probability >= self.regime_halt_probability
        ):
            reasons.append(
                f"crisis regime halt: P(crisis)={state.crisis_probability:.2f} "
                f">= {self.regime_halt_probability}"
            )

        return RiskDecision(allowed=not reasons, reasons=reasons)

    def regime_size_multiplier(self, crisis_probability: float) -> float:
        """1.0 below scale_start, linear ramp to 0.0 at halt_probability."""
        if not self.regime_scaling_enabled:
            return 1.0
        p = max(0.0, min(1.0, crisis_probability))
        if p <= self.regime_scale_start:
            return 1.0
        if p >= self.regime_halt_probability:
            return 0.0
        span = self.regime_halt_probability - self.regime_scale_start
        return 1.0 - (p - self.regime_scale_start) / span

    def check_reward_risk(self, entry: float, stop: float, target: float) -> bool:
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return False
        return abs(target - entry) / stop_distance >= self.min_reward_risk

    def position_size(self, equity: float, risk_percent: float, stop_distance: float) -> float:
        """size = equity * risk_percent / stop_distance, capped at max_risk_per_trade."""
        if stop_distance <= 0:
            raise ValueError("stop_distance must be positive")
        if equity <= 0:
            raise ValueError("equity must be positive")
        risk_percent = min(risk_percent, self.max_risk_per_trade)
        return equity * risk_percent / stop_distance
