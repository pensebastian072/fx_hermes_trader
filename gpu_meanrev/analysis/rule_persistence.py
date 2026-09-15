"""D01 — does a rule that worked recently keep working? (rank-persistence)

WHAT QUESTION THIS ANSWERS
--------------------------
The standing user hypothesis is: "maybe something works for six months or a
year and then it changes — so watch the market early in the period, work out
the rhythm, then trade it, and re-do that each period."

That design is walk-forward re-selection with a burn-in. Whether it can work at
all rests on ONE load-bearing assumption: that a rule which ranked well in the
recent past still ranks well in the near future. If combo rankings are
reshuffled between periods, then adaptive re-selection has nothing to select
on, and no amount of engineering around the selection step rescues it — you
would be picking next period's rule by reading noise.

This module tests that assumption directly, on results that ALREADY EXIST
(B01's 160-combo x 15-year checkpoint). It is a DIAGNOSTIC, not a battery:

  * it selects no rule to trade and proposes no strategy
  * it therefore appends NOTHING to the trial ledger and charges no deflation
  * it touches no holdout

That distinction matters. Running an adaptive battery first and reading its
result would cost trials AND confound two questions (is there persistence? and
does my particular adaptive rule exploit it?). This separates them, and the
cheap question is answered first.

TWO DESIGNS TESTED
------------------
  A. ACROSS years — rank combos on year Y, trade the winner in year Y+1.
     Answers "does an edge survive into the next year?"
  B. BURN-IN within a year — rank combos on the first `burn_months` of year Y,
     trade the winner over the REST of year Y. This is the user's design as
     stated. Answers "can you learn the rhythm early and trade the remainder?"

BENCHMARKS (a bare adaptive return means nothing on its own)
  * oracle  — the best combo in the traded window, chosen with hindsight. Upper
              bound; unreachable by construction.
  * median  — the median combo in the traded window. What you get by not
              selecting at all.
  * mean    — the average combo. What a blind random pick earns in expectation.
  * fixed   — the single best combo over the WHOLE period, chosen with full
              hindsight and held throughout. Itself optimistic, and included so
              that "adaptive beats fixed" cannot be claimed against a strawman.

THE CAVEAT THAT DECIDES HOW THIS READS
--------------------------------------
B01's stored `pnl` is NET of cost — backtest/batch_walkforward.py:140 books
`position * (px - entry) / entry - 2.0 * cost_per_side` on every exit. So every
level below is net of the 1.0 bp round-trip assumption that
journal/B01_B02_POSTMORTEM.md showed to be ~2.5x the entire gross edge of
0.406 bps.

An earlier version of this docstring claimed the RANK findings were robust to
that assumption, on the reasoning that a roughly combo-constant per-round-trip
cost shifts levels without reshuffling order. THAT IS FALSE AND IT IS THE MOST
IMPORTANT THING ON THIS PAGE. The per-round-trip RATE is constant; total cost is
rate x trade count, and trade count spans ~321x across this grid. So cost does
not shift the ranking, it INVERTS it — measured same-year Spearman between net
and gross totals is +0.16 (2009), -0.63 (2015), -0.61 (2022), while gross ranks
+0.83..+0.91 with turnover and net ranks -0.85..-0.93.

The rank findings are therefore the MOST cost-sensitive output here, not the
least. Concretely: ranking combos by net PnL under a 1 bp charge is close to
ranking them by "trades least", which requires no market knowledge at all. The
`turnover_control` section exists to separate the two, and it is the section to
read first — the headline Spearman without it is not a persistence result.

PROVENANCE
----------
B01's year files 2008-2022 are the clean rerun (commit 2257c41). The archive's
wrong-scale bars all sit in 2000-2005 (journal/scorecards/data_provenance.json),
i.e. entirely outside this window — verified, not assumed, per CLAUDE.md rule 4.
The quarantined pre-fix blocks are NOT read by this module.

COMMAND
-------
  .venv\\Scripts\\python.exe -m gpu_meanrev.analysis.rule_persistence
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_meanrev import config
from gpu_meanrev.experiments import registry

CHECKPOINT = config.DATA_REPORTS / "B01_zscore_reversion_checkpoint" / "trades"
DIAGNOSTIC_ID = "D01_rule_persistence"

BURN_MONTHS = [3, 6]
LAG_YEARS = [1, 2, 3]


def load_year(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """((day x combo) net PnL matrix, per-combo summary) for one year.

    Day-aggregated on purpose (CLAUDE.md rule 2): B01's trades overlap heavily
    and run across correlated pairs, so per-trade rows overstate n by ~27x
    (the checkpoint's own day_cluster_ratio).

    The summary carries the columns needed to tell a real edge from a turnover
    artifact: trade count, and GROSS PnL reconstructed by adding back the
    round-trip charge that batch_walkforward.py:140 subtracted. Without those,
    a net-PnL ranking cannot be distinguished from a "trades least" ranking.
    """
    path = CHECKPOINT / f"year_{year}.parquet"
    df = pd.read_parquet(path, columns=["ts", "combo_idx", "pnl", "pair"])
    df["day"] = df["ts"].dt.date
    daily = df.groupby(["day", "combo_idx"])["pnl"].sum().unstack("combo_idx").sort_index()

    charge = df["pair"].map(config.cost_per_side).astype(float) * 2.0
    df["gross"] = df["pnl"] + charge
    summary = df.groupby("combo_idx").agg(
        net_total=("pnl", "sum"),
        gross_total=("gross", "sum"),
        n_trades=("pnl", "size"),
    )
    summary["per_trade_gross_bps"] = summary["gross_total"] / summary["n_trades"] * 1e4
    return daily, summary


def available_years() -> list[int]:
    years = sorted(int(p.stem.split("_")[1]) for p in CHECKPOINT.glob("year_*.parquet"))
    # CLAUDE.md rule 6: a diagnostics path that reimplements loading must
    # re-assert the holdout clip. This module reads year files off disk rather
    # than through build_panel, so nothing else stops a future year_2023.parquet
    # from being globbed in while the scorecard still prints holdout_spent=false.
    holdout_year = int(config.HOLDOUT_START[:4])
    leaked = [y for y in years if y >= holdout_year]
    if leaked and not config.HOLDOUT_UNLOCK:
        raise RuntimeError(
            f"checkpoint contains holdout years {leaked} (>= {holdout_year}); "
            "this diagnostic is in-sample only"
        )
    return years


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (not a dependency of this repo)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _bench(totals: pd.Series, picked: int) -> dict:
    """One traded window: what the pick earned vs what the alternatives earned."""
    return {
        "picked_combo": int(picked),
        "picked": float(totals.get(picked, np.nan)),
        "oracle": float(totals.max()),
        "median": float(totals.median()),
        "mean": float(totals.mean()),
        "picked_pctile": float((totals < totals.get(picked, np.nan)).mean()),
    }


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank correlation of x and y with z partialled out."""
    rxy, rxz, ryz = _spearman(x, y), _spearman(x, z), _spearman(y, z)
    denom = np.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    return float((rxy - rxz * ryz) / denom)


def turnover_control(summaries: dict[int, pd.DataFrame]) -> dict:
    """Is the year-to-year rank persistence an EDGE, or just turnover?

    Three lag-1 rank correlations, side by side:
      * net_total            -- what the headline reports
      * net_total | n_trades -- the same thing with turnover partialled out
      * per_trade_gross_bps  -- scale-free edge QUALITY, immune to trade count
    plus n_trades' own persistence, which sets the ceiling on how much of the
    first number the third has to explain.

    Then the decisive test: select on scale-free edge quality in year Y and see
    what percentile that pick lands in on year Y+1's edge quality. If the
    headline persistence were an edge, this should look like the headline's
    ~84th percentile. If it were turnover, this should look like a coin flip.
    """
    years = sorted(summaries)
    series = {"net_total": [], "net_given_trades": [], "per_trade_gross_bps": [], "n_trades": []}
    picks = []

    for y0, y1 in zip(years[:-1], years[1:]):
        a, b = summaries[y0], summaries[y1]
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]

        series["net_total"].append(_spearman(a["net_total"].to_numpy(), b["net_total"].to_numpy()))
        series["n_trades"].append(_spearman(a["n_trades"].to_numpy(), b["n_trades"].to_numpy()))
        series["per_trade_gross_bps"].append(
            _spearman(a["per_trade_gross_bps"].to_numpy(), b["per_trade_gross_bps"].to_numpy())
        )
        series["net_given_trades"].append(_partial_spearman(
            a["net_total"].to_numpy(), b["net_total"].to_numpy(), a["n_trades"].to_numpy()
        ))

        picked = int(a["per_trade_gross_bps"].idxmax())
        q = b["per_trade_gross_bps"]
        picks.append({
            "select_year": y0, "trade_year": y1, "picked_combo": picked,
            "picked_pctile_on_edge_quality": float((q < q.loc[picked]).mean()),
            "picked_n_trades": int(a.loc[picked, "n_trades"]),
        })

    net_vs_gross = {
        int(y): round(_spearman(s["net_total"].to_numpy(), s["gross_total"].to_numpy()), 4)
        for y, s in summaries.items()
    }
    trade_range = {
        int(y): {"min": int(s["n_trades"].min()), "max": int(s["n_trades"].max()),
                 "ratio": round(float(s["n_trades"].max() / max(1, s["n_trades"].min())), 1)}
        for y, s in summaries.items()
    }

    return {
        "lag1_mean_spearman": {k: round(float(np.mean(v)), 4) for k, v in series.items()},
        "lag1_min_spearman": {k: round(float(np.min(v)), 4) for k, v in series.items()},
        "edge_quality_selection": {
            "picks": picks,
            "mean_pctile": round(float(np.mean([p["picked_pctile_on_edge_quality"] for p in picks])), 4),
            "beat_median_years": int(sum(1 for p in picks if p["picked_pctile_on_edge_quality"] > 0.5)),
            "n": len(picks),
        },
        "same_year_spearman_net_vs_gross": net_vs_gross,
        "trade_count_range_per_year": trade_range,
        "reading": (
            "If lag1(net_total) is high while lag1(net|n_trades) and "
            "lag1(per_trade_gross_bps) are much lower, and edge-quality selection "
            "lands near the 50th percentile, then the headline persistence is a "
            "turnover ranking under a fixed per-trade charge, not a persistent edge."
        ),
    }


def across_years(yearly: dict[int, pd.DataFrame]) -> dict:
    """Design A: select on year Y, trade year Y+1."""
    years = sorted(yearly)
    totals = {y: yearly[y].sum(axis=0) for y in years}

    transitions = []
    for y0, y1 in zip(years[:-1], years[1:]):
        a, b = totals[y0], totals[y1]
        common = a.index.intersection(b.index)
        picked = int(a.loc[common].idxmax())
        row = {"select_year": y0, "trade_year": y1,
               "spearman": round(_spearman(a.loc[common].to_numpy(), b.loc[common].to_numpy()), 4)}
        row.update(_bench(b.loc[common], picked))
        transitions.append(row)

    lag_corr = {}
    for h in LAG_YEARS:
        vals = []
        for y0 in years:
            y1 = y0 + h
            if y1 not in totals:
                continue
            a, b = totals[y0], totals[y1]
            common = a.index.intersection(b.index)
            vals.append(_spearman(a.loc[common].to_numpy(), b.loc[common].to_numpy()))
        lag_corr[f"lag_{h}y"] = {
            "n": len(vals),
            "mean_spearman": round(float(np.mean(vals)), 4) if vals else None,
        }

    # "fixed": best combo over the whole period, full hindsight, held throughout.
    all_total = sum(totals.values())
    fixed_combo = int(all_total.idxmax())
    traded_years = [t["trade_year"] for t in transitions]
    fixed_sum = float(sum(totals[y].get(fixed_combo, np.nan) for y in traded_years))

    picked_sum = float(sum(t["picked"] for t in transitions))
    return {
        "transitions": transitions,
        "lag_correlation": lag_corr,
        "totals": {
            "adaptive": round(picked_sum, 6),
            "oracle": round(float(sum(t["oracle"] for t in transitions)), 6),
            "median": round(float(sum(t["median"] for t in transitions)), 6),
            "mean": round(float(sum(t["mean"] for t in transitions)), 6),
            "fixed_full_hindsight": round(fixed_sum, 6),
            "fixed_combo": fixed_combo,
        },
        "adaptive_beat_median_years": int(sum(1 for t in transitions if t["picked"] > t["median"])),
        "n_transitions": len(transitions),
        "mean_picked_pctile": round(float(np.mean([t["picked_pctile"] for t in transitions])), 4),
    }


def burn_in(yearly: dict[int, pd.DataFrame], burn_months: int) -> dict:
    """Design B (the user's): rank on the first `burn_months` of a year, trade the rest."""
    rows = []
    for year, daily in sorted(yearly.items()):
        idx = pd.to_datetime(pd.Series(daily.index))
        is_burn = (idx.dt.month <= burn_months).to_numpy()
        if is_burn.all() or not is_burn.any():
            continue
        sel = daily.loc[is_burn].sum(axis=0)
        trade = daily.loc[~is_burn].sum(axis=0)
        picked = int(sel.idxmax())
        row = {"year": int(year),
               "burn_days": int(is_burn.sum()), "trade_days": int((~is_burn).sum()),
               "spearman_burn_vs_rest": round(_spearman(sel.to_numpy(), trade.to_numpy()), 4)}
        row.update(_bench(trade, picked))
        rows.append(row)

    return {
        "burn_months": burn_months,
        "years": rows,
        "totals": {
            "adaptive": round(float(sum(r["picked"] for r in rows)), 6),
            "oracle": round(float(sum(r["oracle"] for r in rows)), 6),
            "median": round(float(sum(r["median"] for r in rows)), 6),
            "mean": round(float(sum(r["mean"] for r in rows)), 6),
        },
        "adaptive_beat_median_years": int(sum(1 for r in rows if r["picked"] > r["median"])),
        "n_years": len(rows),
        "mean_spearman": round(float(np.mean([r["spearman_burn_vs_rest"] for r in rows])), 4),
        "mean_picked_pctile": round(float(np.mean([r["picked_pctile"] for r in rows])), 4),
    }


def main() -> None:  # pragma: no cover - operator entry point
    years = available_years()
    print(f"loading {len(years)} year files: {years[0]}-{years[-1]}")
    yearly, summaries = {}, {}
    for y in years:
        yearly[y], summaries[y] = load_year(y)
        print(f"  {y}: {yearly[y].shape[0]} days x {yearly[y].shape[1]} combos")

    payload = {
        "diagnostic_id": DIAGNOSTIC_ID,
        "source": "B01_zscore_reversion_checkpoint (clean 2008-2022 rerun, 2257c41)",
        "is_a_battery": False,
        "trials_charged": 0,
        # Derived, not asserted: available_years() raises if any file reaches
        # HOLDOUT_START, so reaching this line proves the claim.
        "holdout_spent": bool(max(years) >= int(config.HOLDOUT_START[:4])),
        "n_combos": int(yearly[years[0]].shape[1]),
        "turnover_control": turnover_control(summaries),
        "pnl_is_net_of_cost": True,
        "cost_caveat": (
            "All levels are NET of the 1.0 bp round-trip assumption "
            "(batch_walkforward.py:140), which B01_B02_POSTMORTEM.md showed is "
            "~2.5x the 0.406 bps gross edge. The RANK findings are the most "
            "cost-sensitive output here, NOT the least: total cost is rate x trade "
            "count, trade count spans ~321x across this grid, and net-vs-gross "
            "same-year rank correlation is negative in most years. Read "
            "`turnover_control` before quoting any Spearman from `across_years`."
        ),
        "across_years": across_years(yearly),
    }
    for bm in BURN_MONTHS:
        payload[f"burn_in_{bm}m"] = burn_in(yearly, bm)

    out = registry.write_scorecard(DIAGNOSTIC_ID, payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nscorecard: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
