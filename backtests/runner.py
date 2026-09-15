"""Backtest runner (PROJECT_PLAN.md Phase 7, 29.2, 29.4).

Replays historical bars through the regime detector + regime-gated ensemble +
risk engine + a simulated position, with spread and slippage costs.

No lookahead by construction:
- the HMM is fitted once on the warmup window only;
- per-bar regime probabilities are forward-filtered (past data only);
- specialist scores at bar t see closes[: t + 1];
- entries execute at the NEXT bar's open, stops/targets on bar high/low.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engines.ensemble import RegimeGatedEnsemble
from engines.regime_detection.hmm_regime import CRISIS, HMMRegimeDetector
from risk.risk_engine import PortfolioState, RiskEngine


@dataclass
class BacktestConfig:
    starting_capital: float = 10000.0
    risk_per_trade: float = 0.0025
    # Costs are FRACTIONS of price so they scale across pairs: 0.02 absolute
    # was ~2 pips on USDJPY at 150 but ~300 pips on EURUSD at 1.08, which
    # made every non-JPY trade start far beyond its own stop.
    spread: float = 0.0001  # fraction of price, paid on entry (~1 pip FX)
    slippage: float = 0.00005  # fraction of price, adverse on entry
    long_threshold: float = 70.0
    short_threshold: float = -70.0
    warmup_bars: int = 150
    stop_atr_like_pct: float = 0.004  # stop distance as fraction of price
    reward_risk: float = 2.0
    regime_update_interval: int = 5
    max_window: int = 500  # cap regime-filter window for speed


@dataclass
class Trade:
    entry_index: int
    exit_index: int
    direction: str
    entry: float
    exit: float
    size: float
    pnl: float


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    crisis_halts: int = 0
    risk_vetoes: int = 0
    # bars spent at/above the crisis halt probability - protection can engage
    # even when no signal fires, because the ensemble scores ~0 in crisis
    crisis_halt_bars: int = 0

    @property
    def trade_pnls(self) -> list[float]:
        return [t.pnl for t in self.trades]


def compute_metrics(result: BacktestResult, starting_capital: float) -> dict:
    pnls = np.array(result.trade_pnls)
    equity = np.array(result.equity_curve)
    metrics = {
        "n_trades": int(len(pnls)),
        "net_pnl": float(pnls.sum()) if len(pnls) else 0.0,
        "win_rate": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "crisis_halts": result.crisis_halts,
        "risk_vetoes": result.risk_vetoes,
        "crisis_halt_bars": result.crisis_halt_bars,
    }
    gross_win = pnls[pnls > 0].sum() if len(pnls) else 0.0
    gross_loss = -pnls[pnls < 0].sum() if len(pnls) else 0.0
    if gross_loss > 0:
        metrics["profit_factor"] = float(gross_win / gross_loss)
    elif gross_win > 0:
        metrics["profit_factor"] = float("inf")
    if len(equity) > 1:
        peaks = np.maximum.accumulate(equity)
        metrics["max_drawdown_pct"] = float(((peaks - equity) / peaks).max())
        rets = np.diff(equity) / equity[:-1]
        if rets.std() > 0:
            metrics["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(252))
    metrics["return_pct"] = (
        float((equity[-1] - starting_capital) / starting_capital) if len(equity) else 0.0
    )
    return metrics


class BacktestRunner:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk_engine: RiskEngine | None = None,
        ensemble: RegimeGatedEnsemble | None = None,
        detector: HMMRegimeDetector | None = None,
    ):
        self.config = config or BacktestConfig()
        self.risk_engine = risk_engine or RiskEngine(
            {"starting_capital": self.config.starting_capital,
             "regime_scaling": {"enabled": True}}
        )
        self.ensemble = ensemble or RegimeGatedEnsemble()
        self.detector = detector or HMMRegimeDetector()

    def run(self, df: pd.DataFrame) -> BacktestResult:
        cfg = self.config
        closes = df["close"].to_numpy(dtype=float)
        if len(closes) <= cfg.warmup_bars + 10:
            raise ValueError("not enough bars for warmup")

        self.detector.fit(closes[: cfg.warmup_bars])

        result = BacktestResult()
        equity = cfg.starting_capital
        position: dict | None = None
        regime_probs = {CRISIS: 0.0}
        pending_direction: str | None = None

        for i in range(cfg.warmup_bars, len(df) - 1):
            bar = df.iloc[i]

            # manage open position on this bar's range
            if position is not None:
                exit_price = None
                if position["direction"] == "long":
                    if bar["low"] <= position["stop"]:
                        exit_price = position["stop"]
                    elif bar["high"] >= position["target"]:
                        exit_price = position["target"]
                else:
                    if bar["high"] >= position["stop"]:
                        exit_price = position["stop"]
                    elif bar["low"] <= position["target"]:
                        exit_price = position["target"]
                if exit_price is not None:
                    sign = 1 if position["direction"] == "long" else -1
                    pnl = sign * (exit_price - position["entry"]) * position["size"]
                    equity += pnl
                    result.trades.append(
                        Trade(
                            entry_index=position["entry_index"],
                            exit_index=i,
                            direction=position["direction"],
                            entry=position["entry"],
                            exit=exit_price,
                            size=position["size"],
                            pnl=pnl,
                        )
                    )
                    position = None

            # execute signal queued on the previous bar at THIS bar's open
            if pending_direction is not None and position is None:
                direction = pending_direction
                sign = 1 if direction == "long" else -1
                open_price = float(bar["open"])
                entry = open_price + sign * open_price * (cfg.spread + cfg.slippage)
                stop_distance = entry * cfg.stop_atr_like_pct
                size = self.risk_engine.position_size(
                    equity, cfg.risk_per_trade, stop_distance
                )
                size *= self.risk_engine.regime_size_multiplier(regime_probs.get(CRISIS, 0.0))
                if size > 0:
                    position = {
                        "direction": direction,
                        "entry": entry,
                        "stop": entry - sign * stop_distance,
                        "target": entry + sign * cfg.reward_risk * stop_distance,
                        "size": size,
                        "entry_index": i,
                    }
            pending_direction = None

            # regime + score using data up to and including bar i only
            window_start = max(0, i + 1 - cfg.max_window)
            window = closes[window_start : i + 1]
            if (i - cfg.warmup_bars) % cfg.regime_update_interval == 0:
                regime_probs = self.detector.regime_probabilities(window)
            if (
                self.risk_engine.regime_scaling_enabled
                and regime_probs.get(CRISIS, 0.0) >= self.risk_engine.regime_halt_probability
            ):
                result.crisis_halt_bars += 1

            score = self.ensemble.score(window, regime_probs)
            direction = None
            if score.final_score >= cfg.long_threshold:
                direction = "long"
            elif score.final_score <= cfg.short_threshold:
                direction = "short"

            if direction is not None and position is None:
                state = PortfolioState(
                    open_trades=0,
                    crisis_probability=regime_probs.get(CRISIS, 0.0),
                    daily_pnl_pct=0.0,
                )
                decision = self.risk_engine.evaluate("BACKTEST", bar["timestamp"], state)
                if decision.allowed:
                    pending_direction = direction
                else:
                    result.risk_vetoes += 1
                    if any("crisis regime halt" in r for r in decision.reasons):
                        result.crisis_halts += 1

            result.equity_curve.append(equity)

        # mark-to-market close of any open position at the last bar
        if position is not None:
            last_close = float(df.iloc[-1]["close"])
            sign = 1 if position["direction"] == "long" else -1
            pnl = sign * (last_close - position["entry"]) * position["size"]
            equity += pnl
            result.trades.append(
                Trade(
                    entry_index=position["entry_index"],
                    exit_index=len(df) - 1,
                    direction=position["direction"],
                    entry=position["entry"],
                    exit=last_close,
                    size=position["size"],
                    pnl=pnl,
                )
            )
            result.equity_curve.append(equity)

        result.metrics = compute_metrics(result, cfg.starting_capital)
        return result


def save_report(result: BacktestResult, name: str, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{name}.json"
    payload = {
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": result.metrics,
        "n_trades": len(result.trades),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
