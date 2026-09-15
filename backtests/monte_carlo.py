"""Monte Carlo + structural macro stress testing (PROJECT_PLAN.md 29.4).

Two layers:

1. Trade-order reshuffling: resample realized trade PnLs to get a drawdown
   distribution instead of a single path.

2. Structural scenarios injected into the price series, calibrated to the
   kind of moves seen at the 2008 yield high-water mark, violent curve
   inversions, and JPY carry-trade unwinds. A scenario run PASSES only if the
   protection actually engaged (crisis halt / risk vetoes / drawdown
   contained) - surviving because breakers never fired is a failed test of
   the risk system.
"""

from dataclasses import dataclass, field

import numpy as np

SCENARIOS = {
    # drift shock per bar, vol multiplier, gap (one-off fractional jump)
    "RATE_SHOCK": {"drift": -0.002, "vol_mult": 3.0, "gap": 0.0},
    "CARRY_UNWIND": {"drift": -0.004, "vol_mult": 4.0, "gap": -0.03},
    "VOL_EXPLOSION": {"drift": 0.0, "vol_mult": 5.0, "gap": 0.0},
}


def reshuffle_drawdowns(
    trade_pnls: list[float],
    starting_capital: float = 10000.0,
    n_sims: int = 1000,
    seed: int = 7,
) -> dict:
    """Reshuffle trade order; return max-drawdown distribution percentiles."""
    pnls = np.asarray(trade_pnls, dtype=float)
    if len(pnls) == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n_sims": 0}
    rng = np.random.default_rng(seed)
    max_dds = np.empty(n_sims)
    for s in range(n_sims):
        equity = starting_capital + np.cumsum(rng.permutation(pnls))
        equity = np.concatenate([[starting_capital], equity])
        peaks = np.maximum.accumulate(equity)
        max_dds[s] = ((peaks - equity) / peaks).max()
    return {
        "p50": float(np.percentile(max_dds, 50)),
        "p95": float(np.percentile(max_dds, 95)),
        "p99": float(np.percentile(max_dds, 99)),
        "n_sims": n_sims,
    }


def apply_scenario(
    closes: np.ndarray, scenario: str, start_index: int, duration: int, seed: int = 7
) -> np.ndarray:
    """Inject a structural shock into a close series from start_index onward."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario}; choose from {list(SCENARIOS)}")
    params = SCENARIOS[scenario]
    closes = np.asarray(closes, dtype=float).copy()
    rng = np.random.default_rng(seed)

    end = min(start_index + duration, len(closes))
    base_returns = np.diff(np.log(closes))
    base_vol = base_returns.std() if len(base_returns) else 0.001

    shocked = closes[:start_index].tolist()
    price = closes[start_index - 1] if start_index > 0 else closes[0]
    price *= 1.0 + params["gap"]  # gap hits immediately
    for i in range(start_index, end):
        ret = params["drift"] + rng.normal(0, base_vol * params["vol_mult"])
        price *= np.exp(ret)
        shocked.append(price)
    # after the shock window, resume original relative moves from the new level
    for i in range(end, len(closes)):
        ret = np.log(closes[i] / closes[i - 1])
        price *= np.exp(ret)
        shocked.append(price)
    return np.asarray(shocked)


@dataclass
class StressReport:
    scenario: str
    metrics: dict = field(default_factory=dict)
    breakers_engaged: bool = False
    passed: bool = False


def run_stress_scenario(
    closes: np.ndarray,
    scenario: str,
    start_index: int,
    duration: int,
    max_drawdown_limit: float = 0.08,
    seed: int = 7,
) -> StressReport:
    """Shock the series, run a backtest, verify protection engaged."""
    from backtests.data_loader import closes_to_ohlc
    from backtests.runner import BacktestRunner

    shocked = apply_scenario(closes, scenario, start_index, duration, seed=seed)
    result = BacktestRunner().run(closes_to_ohlc(shocked))

    engaged = (
        result.crisis_halts > 0 or result.risk_vetoes > 0 or result.crisis_halt_bars > 0
    )
    contained = result.metrics["max_drawdown_pct"] <= max_drawdown_limit
    return StressReport(
        scenario=scenario,
        metrics=result.metrics,
        breakers_engaged=engaged,
        passed=engaged and contained,
    )
