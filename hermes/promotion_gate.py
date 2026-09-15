"""Promotion gate: deterministic checks a strategy config must pass before
Hermes may promote it to paper trading (PROJECT_PLAN.md section 12).

Gates come from configs/active/autonomy.yaml promotion_gates. This module
only evaluates and reports - it never edits configs. Live promotion is not
implemented anywhere and always requires human approval by policy.
"""

from dataclasses import dataclass, field

from app.services.config_service import load_config


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)


def evaluate_backtest_gates(metrics: dict, gates: dict | None = None) -> GateResult:
    """Check a backtest metrics dict (backtests.runner.compute_metrics output)."""
    gates = gates if gates is not None else (
        load_config("autonomy").get("promotion_gates") or {}
    )
    failures: list[str] = []
    checked: list[str] = []

    min_pf = gates.get("min_profit_factor")
    if min_pf is not None:
        checked.append("min_profit_factor")
        pf = float(metrics.get("profit_factor", 0.0))
        if pf < float(min_pf):
            failures.append(f"profit_factor {pf:.2f} < required {float(min_pf):.2f}")

    max_dd = gates.get("max_backtest_drawdown")
    if max_dd is not None:
        checked.append("max_backtest_drawdown")
        dd = float(metrics.get("max_drawdown_pct", 1.0))
        if dd > float(max_dd):
            failures.append(f"max_drawdown {dd:.3f} > allowed {float(max_dd):.3f}")

    if not metrics.get("n_trades"):
        failures.append("backtest produced no trades; nothing to promote")

    return GateResult(passed=not failures, failures=failures, checked=checked)


def evaluate_walk_forward_gates(
    split_metrics: list[dict], gates: dict | None = None
) -> GateResult:
    """Check walk-forward results: enough splits, positive out-of-sample PnL."""
    gates = gates if gates is not None else (
        load_config("autonomy").get("promotion_gates") or {}
    )
    failures: list[str] = []
    checked: list[str] = []

    min_splits = int(gates.get("min_walk_forward_splits", 0))
    if min_splits:
        checked.append("min_walk_forward_splits")
        if len(split_metrics) < min_splits:
            failures.append(
                f"only {len(split_metrics)} walk-forward splits, need {min_splits}"
            )

    if gates.get("require_positive_oos"):
        checked.append("require_positive_oos")
        total_oos = sum(float(m.get("net_pnl", 0.0)) for m in split_metrics)
        if total_oos <= 0:
            failures.append(f"out-of-sample net PnL not positive ({total_oos:.2f})")

    return GateResult(passed=not failures, failures=failures, checked=checked)
