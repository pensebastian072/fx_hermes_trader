"""Post-hoc DIAGNOSTICS on the finished B02 trade log. No new search.

Framing (quant-research-gate rule): everything in this module is a
DESCRIPTION of an already-completed, already-rejected battery. No combo is
selected here, no n_trials is charged, and nothing written here may be used
to pick a strategy — B02's verdict (FAIL, hypothesis rejected) is fixed and
these numbers cannot change it. Their only job is to answer "what does the
0.4bp gross edge actually consist of, and what would have to be true for it
to matter?" so that the NEXT pre-registration is informed rather than blind.

Questions it answers, none of which B02 itself asked:
  Q1 cost breakeven — at what round-trip cost does the control arm turn
     positive, and how far is that from the 1.0bp the study assumed?
  Q2 exit-hour profile — is the gross edge spread across the clock or
     concentrated in the illiquid hours where the assumed flat spread is
     least defensible?
  Q3 per-pair x per-year — is the pooled decay a universe-wide fact or a
     couple of pairs dragging an otherwise-flat panel?
  Q4 parameter-surface shape — does gross edge vary smoothly with lookback
     and entry_z (a signal) or look like noise (a fluke)?
  Q5 effective sample size — the gate charged n = one trade per row, but
     trades cluster within days and across 7 correlated pairs. Redo the
     headline Sharpe/DSR on DAILY aggregated PnL, where the observations are
     much closer to independent.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.analysis.b02_diagnostics
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from gpu_meanrev import config, gate
from gpu_meanrev.batteries.b02_mtf_confluence import BATTERY_ID, GRID_B02
from gpu_meanrev.signals.mean_reversion import combo_grid

TRADES_DIR = config.DATA_REPORTS / f"{BATTERY_ID}_checkpoint" / "trades"
OUT_PATH = config.SCORECARDS / f"{BATTERY_ID}_diagnostics.json"

# All 7 B02 pairs are majors, so every trade carries the same assumed cost.
ROUND_TRIP_COST = 2.0 * config.COST_PER_SIDE["major"]  # 1.0 bp of price

# Cost levels to sweep, as ROUND-TRIP fraction of price. 0 = the pure signal,
# 1.0bp = what B02 assumed, 3.0bp = a wide retail major spread.
COST_GRID_BP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def load_trades() -> pd.DataFrame:
    parts = sorted(TRADES_DIR.glob("year_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no B02 trade log under {TRADES_DIR}")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # Recover GROSS pnl: the state machine booked net = gross - round_trip.
    df["gross"] = df["pnl"] + ROUND_TRIP_COST
    return df


def combo_frame() -> pd.DataFrame:
    combos = combo_grid(GRID_B02)
    return pd.DataFrame(
        [{"combo_idx": i, **c.as_dict()} for i, c in enumerate(combos)]
    ).set_index("combo_idx")


def _bps(x: pd.Series) -> float:
    return round(float(x.mean()) * 1e4, 4)


def q1_cost_breakeven(ctrl: pd.DataFrame) -> dict:
    """Net edge and PF of the control arm across a sweep of cost assumptions."""
    g = ctrl["gross"].to_numpy()
    rows = []
    for bp in COST_GRID_BP:
        net = g - bp * 1e-4
        rows.append({
            "round_trip_bp": bp,
            "net_bps": round(float(net.mean()) * 1e4, 4),
            "profit_factor": round(gate.profit_factor(net), 4),
            "sharpe": round(gate.sharpe(net), 6),
        })
    gross_bps = float(g.mean()) * 1e4
    return {
        "assumed_round_trip_bp": ROUND_TRIP_COST * 1e4,
        "gross_edge_bps": round(gross_bps, 4),
        "breakeven_round_trip_bp": round(gross_bps, 4),  # net=0 exactly at cost==gross
        "sweep": rows,
    }


def q2_exit_hour(ctrl: pd.DataFrame) -> list[dict]:
    """NOTE: `ts` is the EXIT bar, not the entry — the trade log never stored
    an entry timestamp (instrumentation gap, see the report's `gaps` field).
    Read this as "when did the position close", which for max_hold combos is
    a fixed offset from entry and for inner_band combos is not."""
    h = ctrl.assign(hour=ctrl["ts"].dt.hour).groupby("hour")
    return [
        {"hour_utc": int(k), "n_trades": int(len(v)),
         "gross_bps": _bps(v["gross"]), "net_bps": _bps(v["pnl"])}
        for k, v in h
    ]


# Hour buckets, UTC, by liquidity character. 13-15 is the session_mask's own
# london_ny_overlap definition (13:00-16:00). 21-02 spans the daily rollover
# and the Asia handover — the thinnest, widest-spread hours of the FX day.
HOUR_BUCKETS = {
    "21-02 rollover/Asia": [21, 22, 23, 0, 1, 2],
    "07-12 London": [7, 8, 9, 10, 11, 12],
    "13-15 LDN/NY overlap": [13, 14, 15],
    "other (03-06, 16-20)": [3, 4, 5, 6, 16, 17, 18, 19, 20],
}


def q2b_hour_bucket_by_year(ctrl: pd.DataFrame) -> dict:
    """Hour-bucket x year gross edge, PnL share, and per-bucket breakeven cost.

    THIS IS AN EXIT-HOUR VIEW. The trade log stores no entry timestamp for
    B01/B02 (fixed for future runs, see backtest/batch_walkforward.py), and
    the headline combo holds for hours — so a bucket says when a position
    CLOSED, not when it was open or where its spread was paid. It cannot
    support "the edge is earned in the thin hours"; it supports "the PnL is
    booked on closes in the thin hours", which is weaker and is how the
    postmortem must state it.

    Breakeven cost = the bucket's own gross edge, since net is zero exactly
    where cost equals gross. Expressed additionally in EURUSD-equivalent pips
    (1 pip = 0.0001 of price; at a 1.10 handle that is 0.909 bps of price) —
    a rough unit conversion for a mixed-handle basket, not a per-pair number.
    """
    bps_per_pip = 0.0001 / 1.10 * 1e4
    hour_to_bucket = {h: b for b, hs in HOUR_BUCKETS.items() for h in hs}
    d = ctrl.assign(bucket=ctrl["ts"].dt.hour.map(hour_to_bucket))

    total_gross = float(d["gross"].sum())
    rows = []
    for bucket in HOUR_BUCKETS:
        v = d[d["bucket"] == bucket]
        by_year = {int(y): _bps(g["gross"]) for y, g in v.groupby("year")}
        gross_bps = _bps(v["gross"])
        rows.append({
            "bucket": bucket,
            "n_trades": int(len(v)),
            "gross_bps": gross_bps,
            # Signed share of total gross PnL. Buckets sum to 100% by
            # construction; a negative bucket legitimately shows a negative
            # share rather than being folded in by absolute value.
            "share_of_gross_pnl_pct": round(100 * float(v["gross"].sum()) / total_gross, 1),
            "breakeven_round_trip_bp": gross_bps,
            "breakeven_round_trip_pips": round(gross_bps / bps_per_pip, 2),
            "gross_bps_by_year": by_year,
            "breakeven_pips_2022": round(by_year.get(2022, 0.0) / bps_per_pip, 2),
        })
    return {"caveat": "EXIT hour, not entry hour — see docstring", "buckets": rows}


def q3_pair_year(ctrl: pd.DataFrame) -> list[dict]:
    out = []
    for (pair, year), v in ctrl.groupby(["pair", "year"]):
        out.append({
            "pair": pair, "year": int(year), "n_trades": int(len(v)),
            "gross_bps": _bps(v["gross"]),
            "gross_pf": round(gate.profit_factor(v["gross"].to_numpy()), 4),
        })
    return out


def q4_param_surface(ctrl: pd.DataFrame, combos: pd.DataFrame) -> dict:
    j = ctrl.join(combos, on="combo_idx")
    surface = []
    for (lb, ez, ex), v in j.groupby(["lookback_min", "entry_z", "exit_rule"]):
        surface.append({
            "lookback_min": int(lb), "entry_z": float(ez), "exit_rule": ex,
            "n_trades": int(len(v)), "gross_bps": _bps(v["gross"]),
        })
    return {"by_lookback_entryz_exit": surface}


def q5_effective_sample(df: pd.DataFrame, ctrl: pd.DataFrame, combos: pd.DataFrame) -> dict:
    """Redo the headline numbers on DAILY aggregated PnL.

    The gate was handed one observation per trade (n=17368 for the best
    combo). Trades overlap in time and run across 7 USD-correlated pairs, so
    those rows are nowhere near independent, and an overstated n makes the
    Deflated Sharpe LOOK BETTER than it is. B02 failed anyway, which is the
    conservative direction — this quantifies by how much the pooled n was
    flattering it.

    Daily aggregation = the equal-weight portfolio's realised PnL per session
    day: still not perfectly independent, but a day is the natural block here.
    """
    out = {}
    # combo 64 is the battery's headline combo and sits in the 0.67 arm, not
    # the control arm — it has to come from the full log, not `ctrl`.
    subsets = {
        "best_combo_64": df[df["combo_idx"] == 64],
        "control_arm_all": ctrl,
    }
    for label, v in subsets.items():
        per_trade = v["pnl"].to_numpy()
        daily = v.groupby(v["ts"].dt.normalize())["pnl"].mean()
        n_trials = len(combos)
        out[label] = {
            "n_trades": int(len(per_trade)),
            "n_days": int(len(daily)),
            "trades_per_day": round(len(per_trade) / max(len(daily), 1), 2),
            "per_trade": {
                "sharpe": round(gate.sharpe(per_trade), 6),
                "dsr_ratio": round(gate.deflated_sharpe(per_trade, n_trials=n_trials)["ratio"], 4),
            },
            "daily_aggregated": {
                "sharpe": round(gate.sharpe(daily.to_numpy()), 6),
                "dsr_ratio": round(gate.deflated_sharpe(daily.to_numpy(), n_trials=n_trials)["ratio"], 4),
                "mean_daily_bps": round(float(daily.mean()) * 1e4, 4),
                "t_stat": round(float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))), 4),
            },
        }
    return out


def main() -> dict:
    df = load_trades()
    combos = combo_frame()
    ctrl_idx = combos.index[combos["confluence_max"] == 1.01]
    ctrl = df[df["combo_idx"].isin(ctrl_idx)].copy()

    report = {
        "battery_id": BATTERY_ID,
        "note": "post-hoc diagnostics on a CLOSED, REJECTED battery; selects nothing",
        "n_trades_all_arms": int(len(df)),
        "n_trades_control_arm": int(len(ctrl)),
        "q1_cost_breakeven": q1_cost_breakeven(ctrl),
        "q2_exit_hour_utc": q2_exit_hour(ctrl),
        "q2b_hour_bucket_by_year": q2b_hour_bucket_by_year(ctrl),
        "q3_pair_year": q3_pair_year(ctrl),
        "q4_param_surface": q4_param_surface(ctrl, combos),
        "q5_effective_sample": q5_effective_sample(df, ctrl, combos),
        "gaps": [
            "trade log stores EXIT ts only — no entry ts, no side, no hold "
            "duration, no entry z. Hour/side/holding-period attribution is "
            "not recoverable from it and needs re-instrumentation.",
            "entry and exit both execute at the SAME bar's close that "
            "generated the signal — a zero-latency assumption never tested.",
            "cost is a flat 0.5bp/side for every pair, hour and year; real "
            "major spreads vary by session and compressed over 2015-2022.",
        ],
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps({k: v for k, v in rep.items() if k != "q3_pair_year"},
                     indent=2, default=str)[:6000])
    print(f"\nwrote {OUT_PATH}")
