"""Alert processing pipeline.

validate (done by route) -> dedupe -> log alert -> score -> risk check ->
paper order -> decision trace -> structured response.

Every alert produces either a line in alerts.jsonl plus a decision trace,
or a line in rejected_alerts.jsonl. Nothing is hidden.
"""

from datetime import datetime, timezone
from pathlib import Path

from app.schemas.alerts import TradingViewAlert
from app.schemas.decisions import DecisionTrace
from app.services.artifact_service import append_jsonl, read_jsonl
from engines.regime_detection.hmm_regime import load_snapshot
from engines.scoring import ScoringStub
from execution.paper_broker import PaperBroker
from risk.risk_engine import PortfolioState, RiskEngine


class SignalService:
    def __init__(
        self,
        logs_dir: Path,
        risk_engine: RiskEngine,
        broker: PaperBroker,
        scorer: ScoringStub,
        watchlist: list[str],
        mode: str = "paper",
        artifacts_dir: Path | None = None,
    ):
        self.artifacts_dir = artifacts_dir
        self.alerts_path = logs_dir / "alerts.jsonl"
        self.rejected_path = logs_dir / "rejected_alerts.jsonl"
        self.traces_path = logs_dir / "decision_traces.jsonl"
        self.risk_engine = risk_engine
        self.broker = broker
        self.scorer = scorer
        self.watchlist = [s.upper() for s in watchlist]
        self.mode = mode
        # Dedupe keys survive restarts because they are replayed from the alert log.
        self._seen_keys: set[str] = {
            rec["dedupe_key"] for rec in read_jsonl(self.alerts_path) if "dedupe_key" in rec
        }

    def process_alert(self, alert: TradingViewAlert) -> dict:
        key = alert.dedupe_key()
        received_at = datetime.now(timezone.utc).isoformat()

        if key in self._seen_keys:
            self._log_rejected(alert, received_at, "duplicate alert")
            return {
                "status": "duplicate",
                "symbol": alert.symbol,
                "decision": "blocked_duplicate",
                "mode": self.mode,
                "live_trading": False,
            }

        if alert.symbol not in self.watchlist:
            self._log_rejected(alert, received_at, f"symbol {alert.symbol} not in watchlist")
            return {
                "status": "rejected",
                "symbol": alert.symbol,
                "decision": "symbol_not_in_watchlist",
                "mode": self.mode,
                "live_trading": False,
            }

        self._seen_keys.add(key)
        record = alert.model_dump(mode="json")
        record["dedupe_key"] = key
        record["received_at"] = received_at
        append_jsonl(self.alerts_path, record)

        score = self.scorer.score(alert)

        if score.direction == "flat":
            self._write_trace(alert, score, "reject", rejection_reason="score below threshold")
            return {
                "status": "accepted",
                "symbol": alert.symbol,
                "decision": "no_trade",
                "final_score": score.final_score,
                "reason": "score below threshold",
                "mode": self.mode,
                "live_trading": False,
            }

        crisis_probability = 0.0
        if self.artifacts_dir is not None:
            snapshot = load_snapshot(self.artifacts_dir)
            if snapshot is not None:
                crisis_probability = snapshot.crisis_probability

        state = PortfolioState(
            open_trades=len(self.broker.open_positions()),
            trades_today=self.broker.trades_today(),
            trades_today_for_pair=self.broker.trades_today(alert.symbol),
            daily_pnl_pct=self.broker.daily_pnl_pct(),
            weekly_pnl_pct=self.broker.weekly_pnl_pct(),
            crisis_probability=crisis_probability,
        )
        risk = self.risk_engine.evaluate(alert.symbol, alert.timestamp, state)
        if not risk.allowed:
            self._write_trace(
                alert,
                score,
                "reject",
                rejection_reason="; ".join(risk.reasons),
                risk_checks_failed=risk.reasons,
            )
            return {
                "status": "accepted",
                "symbol": alert.symbol,
                "decision": "rejected_by_risk",
                "final_score": score.final_score,
                "reasons": risk.reasons,
                "mode": self.mode,
                "live_trading": False,
            }

        stop_distance = abs(score.entry - score.stop)
        risk_percent = self.risk_engine.base_risk_per_trade
        size = self.risk_engine.position_size(self.broker.equity, risk_percent, stop_distance)
        size *= self.risk_engine.regime_size_multiplier(crisis_probability)
        order = self.broker.create_order(
            symbol=alert.symbol,
            direction=score.direction,
            entry=score.entry,
            stop=score.stop,
            target=score.target,
            size=size,
            risk_percent=risk_percent,
        )
        self._write_trace(
            alert, score, "accept", size=size, risk_percent=risk_percent, order_id=order.order_id
        )
        return {
            "status": "accepted",
            "symbol": alert.symbol,
            "decision": "paper_order_created",
            "final_score": score.final_score,
            "order_id": order.order_id,
            "mode": self.mode,
            "live_trading": False,
        }

    def _log_rejected(self, alert: TradingViewAlert, received_at: str, reason: str) -> None:
        record = alert.model_dump(mode="json")
        record["received_at"] = received_at
        record["rejection_reason"] = reason
        append_jsonl(self.rejected_path, record)

    def _write_trace(
        self,
        alert: TradingViewAlert,
        score,
        decision: str,
        rejection_reason: str | None = None,
        risk_checks_failed: list[str] | None = None,
        size: float | None = None,
        risk_percent: float | None = None,
        order_id: str | None = None,
    ) -> None:
        trace = DecisionTrace(
            timestamp=datetime.now(timezone.utc),
            symbol=alert.symbol,
            strategy=alert.strategy,
            signal=alert.signal,
            direction=score.direction,
            final_score=score.final_score,
            decision=decision,
            reason=score.reason,
            rejection_reason=rejection_reason,
            risk_checks_failed=risk_checks_failed or [],
            entry=score.entry,
            stop=score.stop,
            target=score.target,
            size=size,
            risk_percent=risk_percent,
            order_id=order_id,
            mode=self.mode,
        )
        append_jsonl(self.traces_path, trace.model_dump(mode="json"))
