"""Per-pair table, clustering diagnostic, pooled gate verdict -> scorecard.

SHADOW framing is unconditional here: this module only ever writes a
scorecard + a RESULTS.md row. Nothing in gpu_meanrev/ writes an enforce or
promotion flag, regardless of outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gpu_meanrev import gate
from gpu_meanrev.backtest.bootstrap_ci import confidence
from gpu_meanrev.experiments import registry


def clustering_diagnostic(entry_timestamps: pd.Series) -> dict:
    """Pooled trade count vs. distinct entry-minutes/session-days.

    Same check that caught options_desk's "n=369 pooled -> 41 independent"
    global-entry-rule trap: FX pairs sharing USD moves and session overlap
    make simultaneous cross-pair entries a live risk here. CSCV's block-based
    combinatorics partially guard against this (it segments the pooled
    series into contiguous blocks) but not against *simultaneous* cross-pair
    entries within a block -- report this number standalone, next to the
    pooled verdict, don't bury it.
    """
    n_trades = len(entry_timestamps)
    if n_trades == 0:
        return {"n_trades": 0, "distinct_entry_minutes": 0, "distinct_session_days": 0,
                "minute_cluster_ratio": None, "day_cluster_ratio": None}
    distinct_minutes = entry_timestamps.nunique()
    distinct_days = pd.DatetimeIndex(entry_timestamps).normalize().nunique()
    return {
        "n_trades": n_trades,
        "distinct_entry_minutes": int(distinct_minutes),
        "distinct_session_days": int(distinct_days),
        "minute_cluster_ratio": round(n_trades / distinct_minutes, 3) if distinct_minutes else None,
        "day_cluster_ratio": round(n_trades / distinct_days, 3) if distinct_days else None,
    }


def per_pair_table(pnl_by_pair: dict[str, np.ndarray]) -> list[dict]:
    """One row per pair: n_trades, PF, Sharpe, DSR (n_trials=1 -- a per-pair
    look is not multiplying the search, it's a diagnostic slice)."""
    rows = []
    for pair, pnls in pnl_by_pair.items():
        pnls = np.asarray(pnls, dtype=float)
        pnls = pnls[np.isfinite(pnls)]
        n = len(pnls)
        pf = gate.profit_factor(pnls) if n else None
        sr = gate.sharpe(pnls) if n else None
        dsr = gate.deflated_sharpe(pnls, n_trials=1) if n >= 8 else None
        rows.append({
            "pair": pair, "n_trades": n,
            "profit_factor": pf, "sharpe": sr,
            "dsr_ratio": (dsr or {}).get("ratio"),
            "dsr_prob": (dsr or {}).get("prob"),
        })
    return rows


def pooled_verdict(pooled_pnls: np.ndarray, n_trials: int) -> dict:
    pooled_pnls = np.asarray(pooled_pnls, dtype=float)
    pooled_pnls = pooled_pnls[np.isfinite(pooled_pnls)]
    verdict = gate.evaluate_gate(pooled_pnls, n_trials=n_trials)
    verdict["bootstrap_ci"] = confidence(pooled_pnls)
    return verdict


def build_and_write_scorecard(
    battery_id: str,
    combo_dict: dict,
    pnl_by_pair: dict[str, np.ndarray],
    entry_timestamps: pd.Series,
    n_trials: int,
) -> dict:
    pooled = np.concatenate([np.asarray(v, dtype=float) for v in pnl_by_pair.values()]) if pnl_by_pair else np.array([])
    payload = {
        "battery_id": battery_id,
        "winning_combo": combo_dict,
        "per_pair": per_pair_table(pnl_by_pair),
        "clustering": clustering_diagnostic(entry_timestamps),
        "pooled_verdict": pooled_verdict(pooled, n_trials=n_trials),
        "shadow": True,  # unconditional -- see module docstring
    }
    registry.write_scorecard(battery_id, payload)
    registry.append_result_row(
        battery_id, payload["pooled_verdict"],
        note=f"clustering day_ratio={payload['clustering'].get('day_cluster_ratio')}",
    )
    return payload
