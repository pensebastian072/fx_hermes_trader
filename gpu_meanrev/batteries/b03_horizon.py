"""B03: does the fade pay at a MULTI-HOUR horizon, where cost is amortised?

PRE-REGISTERED 2026-08-06, before any B03 result exists. Registration is
written by `--register-only`, which is meant to be run and committed BEFORE
`--full` is ever invoked.

WHY THIS BATTERY EXISTS
  B01 and B02 both failed, and the 2026-08-06 post-mortem
  (journal/B01_B02_POSTMORTEM.md) says why, in one number: B02's control arm
  gross edge was 0.406 bps against an assumed 1.0 bp round-trip cost. The
  signal was real and roughly 2.5x too small to pay for its own execution.
  That number is B02's, on its 7-major universe — B01's is not comparable
  (its log is 75% crosses on a different cost tier; see the 2026-08-07
  correction in journal/RESULTS.md). B03 inherits B02's universe precisely so
  the comparison stays apples-to-apples.

  The post-mortem also found that the signed payoff after a |z| >= 2 event
  keeps accruing out to 960 minutes with NO SNAP-BACK — no horizon at which
  it peaks and decays. It is not smoothly monotonic (2022 dips at h=480, 2018
  falls away by h=960, 2020 turns negative); the durable part is the absence
  of a reversion peak to exit into. B01/B02 churned through that at
  `max_hold = 4 x lookback`, paying a full round trip for each slice.

HYPOTHESIS
  The per-round-trip gross edge scales with holding horizon while the cost
  per round trip does not. A fade held for hours therefore clears a
  realistic cost hurdle that the same fade held for minutes cannot. B03
  tests whether ANY (lookback, entry_z, hold) combination on hourly bars
  produces a gross edge of at least 2x its assumed round-trip cost, per
  year, on the 7 USD majors.

THIS IS NOT INDEPENDENT EVIDENCE — READ BEFORE QUOTING ANY B03 NUMBER
  The horizon observation was made ON B02's in-sample data, using combo
  64's parameters, which were themselves B02's ex-post best. Running B03
  over 2015-2022 is therefore testing a hypothesis against the data that
  generated it. That is legitimate for DESIGN and illegitimate as PROOF.
  The pre-committed consequence:
    * 2015-2022 is used to fix ONE configuration and to check the kill
      criterion below. Its gate verdict is reported but is NOT evidence of
      an edge, and must be labelled in-sample in RESULTS.md.
    * The only confirmatory test is a SINGLE evaluation of that one frozen
      configuration on the locked 2023+ holdout, run once, at the end,
      and only if the kill criterion is not triggered.
  There is no second holdout. If the holdout fails, B03 is done.

KILL CRITERION (pre-committed; the registered JSON is the sole authority)
  PASS requires: at least one configuration reaches gross edge >= 2.0x its
  assumed round-trip cost in >= 6 of the 8 fold years. Anything else — 5
  years or fewer, for every configuration — is a KILL: B03 is reported as a
  FAIL, and no parameter is added, no universe narrowed, no horizon extended
  to rescue it. This exists so that "the edge is too small to pay for
  execution" cannot be quietly reframed as "we need a wider grid".

  Stated once, here, matching KILL_CRITERION and the registration byte for
  byte. An earlier draft of this docstring said "below 2x in at least 6 of 8
  years" — which kills only at <=2 passing years and disagrees with the
  registered rule for 3, 4 or 5 passing years, exactly the marginal band
  pre-registration exists to close. The registration
  (journal/experiments/registered/B03_horizon.json) is append-only and wins
  any future disagreement.

  `check_kill_criterion()` below computes the verdict mechanically, so the
  rule is machine-CHECKED and not merely machine-readable.

SELECTION RULE for the single configuration taken to the holdout
  Pre-committed here because "use 2015-2022 to fix ONE configuration" names
  no metric, and an unnamed metric can be chosen after seeing the surface:
  take the configuration with the highest IN-SAMPLE DAY-AGGREGATED SHARPE at
  the 1.0 bp cost assumption; ties broken by smaller hold_hours, then smaller
  lookback_hours, then smaller entry_z. Day-aggregated because per-trade n
  overstated the effective sample in B01 and B02 alike.

DESIGN CHOICES, ALL FIXED IN ADVANCE
  * Bar frequency 60 minutes. The post-mortem's finding is about hours; at
    1-minute bars the state machine spends its time on structure the
    hypothesis says is noise, and 1-minute bars were shown to be the wrong
    research frequency for it.
  * Universe = the same 7 USD majors as B02, carried over unchanged on the
    original ex-ante BIS-liquidity ground. Re-picking a universe now, after
    seeing per-pair results, is the p-hacking move this section rules out.
  * Folds are YEARS 2015-2022, reported per year, never pooled into one
    number.
  * Cost is SWEPT, not assumed: every result is reported at 0.5 / 1.0 / 2.0
    bp round-trip, with the gate applied at 1.0 bp (the majors assumption
    B01/B02 used, kept for comparability). Breakeven cost is reported next
    to profit factor, always — the omission that hid B02's real problem.
  * The gate is evaluated on DAY-AGGREGATED PnL, not per-trade rows. B02's
    per-trade n=17,368 overstated an effective sample of ~2,314 days and
    flattered its Deflated Sharpe.

CLI
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b03_horizon --register-only
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b03_horizon --smoke
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b03_horizon --full
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from gpu_meanrev import config, gate
from gpu_meanrev.backtest.batch_walkforward import run_fold
from gpu_meanrev.data.loader import build_panel
from gpu_meanrev.experiments import registry
from gpu_meanrev.features.rolling import rolling_bollinger_percent_b, rolling_zscore
from gpu_meanrev.gpu import get_device, set_seed
from gpu_meanrev.signals.mean_reversion import Combo

BATTERY_ID = "B03_horizon"
DATASET_FAMILY = "fx_1min_meanrev"

# Same 7 USD majors as B02, carried over unchanged (see DESIGN CHOICES).
MAJORS_7 = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

YEARS = list(range(2015, 2023))  # 2023+ stays locked

BAR_MINUTES = 60

# 4 x 3 x 4 = 48 combos. Deliberately SMALLER than B01's 160 and B02's 72:
# every extra trial raises the Deflated Sharpe bar this has to clear, and
# the post-mortem's conclusion is that the binding constraint is cost, not
# insufficient search. Lookbacks and holds are in HOURS.
GRID_B03 = {
    "lookback_hours": [12, 24, 48, 96],
    "entry_z": [1.5, 2.0, 2.5],
    "hold_hours": [6, 24, 72, 168],
    "exit_rule": ["fixed_hold"],
    "session": ["24h"],
    "signal_family": ["zscore_only"],
}

HYPOTHESIS = (
    "Gross edge per round trip scales with holding horizon while cost per "
    "round trip does not, so a z-score fade on 60-minute FX bars held for "
    "hours-to-days clears a realistic cost hurdle that the same fade held "
    "for minutes cannot. Tested on the 7 USD majors, per-year folds "
    "2015-2022, pure price-action. PRIMARY PRE-COMMITTED TEST: at least one "
    "configuration reaches gross edge >= 2x assumed round-trip cost in >= 6 "
    "of 8 fold years; if not, B03 is a FAIL and no grid expansion follows. "
    "In-sample years generated this hypothesis (from B02's horizon curve) "
    "and are therefore design-only; the single confirmatory test is one "
    "evaluation of one frozen configuration on the locked 2023+ holdout."
)

# Pre-committed kill criterion, machine-CHECKED (see check_kill_criterion)
# so it cannot be softened after the fact.
KILL_CRITERION = {
    "metric": "gross_edge_bps / assumed_round_trip_bp",
    "threshold": 2.0,
    "min_years_passing": 6,
    "n_fold_years": 8,
    "scope": "at least one configuration in the 48-combo grid",
    "assumed_round_trip_bp": 1.0,
    # Left implicit in the first draft, which made the rule uncheckable.
    "gross_edge_bps_definition": (
        "for one combo and one fold year: mean over that year's trades of "
        "(net_pnl + assumed_round_trip), pooled across the 7 pairs, expressed "
        "in bps of price (x1e4). All 7 pairs are majors so the added-back cost "
        "is uniform — the tiered-cost error that corrupted B01's headline "
        "gross figure on 2026-08-07 cannot recur on this universe."
    ),
    "selection_rule_for_holdout": (
        "highest in-sample day-aggregated Sharpe at 1.0bp cost; ties to "
        "smaller hold_hours, then lookback_hours, then entry_z"
    ),
    "on_failure": "report FAIL, stop, do not expand grid or narrow universe",
}


def check_kill_criterion(gross_bps_by_combo_year: dict) -> dict:
    """Mechanically evaluate the pre-committed kill criterion.

    `gross_bps_by_combo_year`: {combo_idx: {year: gross_edge_bps}}.

    Returns the verdict plus the per-combo year counts it rests on, so the
    decision is auditable rather than asserted. A rule that only a human
    reads is a rule that can be re-read favourably once the numbers are in;
    this function exists to remove that option.
    """
    thr = KILL_CRITERION["threshold"] * KILL_CRITERION["assumed_round_trip_bp"]
    need = KILL_CRITERION["min_years_passing"]
    per_combo = {}
    for combo_idx, by_year in gross_bps_by_combo_year.items():
        passing = sorted(y for y, v in by_year.items() if v is not None and v >= thr)
        per_combo[combo_idx] = {"years_passing": len(passing), "years": passing}
    best = max(per_combo.items(), key=lambda kv: kv[1]["years_passing"], default=(None, {"years_passing": 0}))
    n_best = best[1]["years_passing"]
    return {
        "threshold_bps": thr,
        "min_years_passing": need,
        "best_combo_idx": best[0],
        "best_years_passing": n_best,
        "passes": n_best >= need,
        "verdict": "PROCEED to single holdout evaluation" if n_best >= need
                   else "KILL — report FAIL, no grid expansion, no universe narrowing",
        "per_combo": per_combo,
    }


def n_combos() -> int:
    n = 1
    for v in GRID_B03.values():
        n *= len(v)
    return n


def b03_combos() -> list[Combo]:
    """The registered 48-combo grid as Combo objects.

    UNITS, because this is where a silent bug would live: B03 runs on HOURLY
    bars, and `Combo.lookback_min` is really "rolling window length in BARS"
    (rolling_zscore takes a bar count, it has never known about minutes). So
    `lookback_hours` maps straight onto it, and `hold_hours` onto `hold_bars`.
    Nothing here is in minutes despite the field name.

    `exit_rule` stays the registered string "fixed_hold"; run_fold treats it
    as the same exit mechanism as "max_hold".
    """
    out = []
    for lb in GRID_B03["lookback_hours"]:
        for ez in GRID_B03["entry_z"]:
            for hold in GRID_B03["hold_hours"]:
                out.append(Combo(lookback_min=lb, entry_z=ez, exit_rule="fixed_hold",
                                 session="24h", signal_family="zscore_only",
                                 hold_bars=hold))
    return out


def hourly_panel(pairs: list[str], year: int) -> pd.DataFrame:
    """1-minute closes -> hourly closes, stamped at the bar's CLOSE.

    `build_panel` enforces the holdout clip, so this inherits it. Resampling
    with label="right"/closed="right" means the bar timestamped 14:00 is the
    price AT 14:00, built from 13:01-14:00 — an entry at that stamp trades a
    price that already exists. Labelling left would stamp it 13:00 and make
    every signal look one hour early, which is the classic resample leak.
    """
    minute = build_panel(pairs, start_date=f"{year}-01-01", end_date=f"{year + 1}-01-01")
    if minute.empty:
        return minute
    return minute.resample("60min", label="right", closed="right").last().dropna(how="all")


def run_year(year: int, pairs: list[str], combos: list[Combo], device) -> pd.DataFrame:
    panel = hourly_panel(pairs, year)
    if panel.empty:
        return pd.DataFrame()
    windows = sorted({c.lookback_min for c in combos})
    if len(panel) < max(windows) + 1:
        return pd.DataFrame()

    prices = torch.tensor(panel.to_numpy(dtype="float32").T)
    z_by = {w: rolling_zscore(prices, w).float() for w in windows}
    bb_by = {w: rolling_bollinger_percent_b(prices, w).float() for w in windows}
    cost = torch.tensor([config.cost_per_side(p) for p in pairs], dtype=torch.float32)

    # A whole year of hourly bars is ~8,760 rows; 48 combos x 7 pairs = 336
    # batch rows. That fits comfortably, so unlike B01/B02 there is no block
    # chunking here — and therefore no position reset at block boundaries.
    _pnl, trades = run_fold(
        prices=prices, z_by_window=z_by, bb_by_window=bb_by,
        fold_index=panel.index, combos=combos, pairs=pairs,
        cost_per_side=cost, device=device, collect_trades=True,
    )
    df = pd.DataFrame(trades)
    if not df.empty:
        df["year"] = year
    return df


def gross_bps_by_combo_year(df: pd.DataFrame) -> dict:
    """{combo_idx: {year: gross_edge_bps}} exactly as KILL_CRITERION defines it.

    All 7 pairs are majors, so the added-back round trip is uniform. That is
    the specific thing that went wrong on B01, whose log is 75% crosses on a
    different cost tier — asserted here rather than assumed.
    """
    assert all(p in config.MAJORS for p in MAJORS_7), "B03 universe must be all majors"
    rt = 2.0 * config.COST_PER_SIDE["major"]
    out: dict = {}
    for (ci, yr), g in df.groupby(["combo_idx", "year"]):
        out.setdefault(int(ci), {})[int(yr)] = float((g["pnl"] + rt).mean()) * 1e4
    return out


def select_for_holdout(df: pd.DataFrame) -> int:
    """The pre-committed selection rule, applied mechanically.

    Highest in-sample DAY-AGGREGATED Sharpe at the 1.0bp cost assumption;
    ties to smaller hold_hours, then lookback_hours, then entry_z.
    """
    combos = b03_combos()
    scored = []
    for ci, g in df.groupby("combo_idx"):
        daily = g.groupby(g["ts"].dt.normalize())["pnl"].mean()
        sr = gate.sharpe(daily.to_numpy()) if len(daily) >= 8 else None
        c = combos[int(ci)]
        scored.append((-(sr if sr is not None else -np.inf), c.max_hold_bars,
                       c.lookback_min, c.entry_z, int(ci)))
    scored.sort()
    return scored[0][4]


def run_battery(years: list[int] | None = None, pairs: list[str] | None = None,
                battery_id: str = BATTERY_ID, publish: bool = True) -> dict:
    """`publish=False` runs the same computation without writing a scorecard or
    a RESULTS.md row — that is what --smoke is for. A partial-year smoke run
    publishing under the real battery id would put a meaningless kill verdict
    into the permanent record (the criterion is defined over all 8 fold years),
    which is the mistake B01/B02 avoid with distinct *_SMOKE ids."""
    years = years or YEARS
    pairs = pairs or MAJORS_7
    device = get_device()
    set_seed(config.SEED)
    t0 = time.time()
    combos = b03_combos()

    ckpt = config.DATA_REPORTS / f"{BATTERY_ID}_checkpoint" / "trades"
    ckpt.mkdir(parents=True, exist_ok=True)
    for year in years:
        out = ckpt / f"year_{year}.parquet"
        if out.exists():
            continue
        run_year(year, pairs, combos, device).to_parquet(out)

    parts = sorted(ckpt.glob("year_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if df.empty:
        return {"battery_id": BATTERY_ID, "error": "no trades"}

    by_cy = gross_bps_by_combo_year(df)
    # The criterion is defined over all 8 fold years. Evaluating it on a subset
    # would return a "KILL" that means only "we didn't run 8 years".
    complete = sorted(years) == sorted(YEARS)
    kill = check_kill_criterion(by_cy) if complete else {
        "passes": False, "best_combo_idx": max(
            by_cy, key=lambda c: sum(v >= 2.0 for v in by_cy[c].values()), default=0),
        "verdict": f"NOT EVALUATED - needs all {len(YEARS)} fold years, ran {len(years)}",
        "best_years_passing": None,
    }

    payload = {
        "battery_id": battery_id, "period": f"{years[0]}-{years[-1]}",
        "kill_criterion_evaluated": complete,
        "in_sample_only": True,
        "framing": ("in-sample years are DESIGN-ONLY (the horizon hypothesis came "
                    "from B02's in-sample data); the confirmatory test is one "
                    "frozen config on the locked 2023+ holdout, run once"),
        "n_trades": int(len(df)), "n_combos": len(combos),
        "kill_criterion": KILL_CRITERION,
        "kill_check": kill,
        "gross_bps_by_combo_year": by_cy,
        "elapsed_sec": round(time.time() - t0, 2),
        "shadow": True,
    }
    if kill["passes"]:
        payload["selected_for_holdout"] = select_for_holdout(df)

    # Gate the best combo on DAY-AGGREGATED PnL (the post-mortem's rule).
    best = kill["best_combo_idx"]
    g = df[df["combo_idx"] == best]
    daily = g.groupby(g["ts"].dt.normalize())["pnl"].mean()
    payload["best_combo"] = combos[best].as_dict()
    payload["day_aggregated_gate"] = gate.evaluate_gate(daily.to_numpy(), n_trials=len(combos))
    payload["n_days"] = int(len(daily))

    if publish:
        registry.write_scorecard(battery_id, payload)
        registry.append_result_row(
            battery_id, payload["day_aggregated_gate"],
            note=f"IN-SAMPLE design pass, NOT evidence; {kill['verdict'][:44]}",
        )
    return payload


def register_only() -> dict:
    """Write the registration WITHOUT running anything.

    Deliberately the only thing this module can do today: the design is
    committed to disk and to git before a single number exists, which is the
    whole point of pre-registration.
    """
    try:
        path = registry.register(
            battery_id=BATTERY_ID, hypothesis=HYPOTHESIS,
            dataset_family=DATASET_FAMILY, pairs=MAJORS_7,
            param_grid={**GRID_B03, "bar_minutes": [BAR_MINUTES],
                        "kill_criterion": [KILL_CRITERION]},
            n_trials=n_combos(),
        )
        return {"registered": str(path), "n_trials": n_combos()}
    except FileExistsError:
        return {"registered": "already", "n_trials": n_combos()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.register_only:
        print(json.dumps(register_only(), indent=2, default=str))
    elif args.smoke:
        out = run_battery(years=[2022], battery_id=f"{BATTERY_ID}_SMOKE", publish=False)
        print(json.dumps({k: v for k, v in out.items()
                          if k != "gross_bps_by_combo_year"}, indent=2, default=str))
    elif args.full:
        out = run_battery()
        print(json.dumps({k: v for k, v in out.items()
                          if k != "gross_bps_by_combo_year"}, indent=2, default=str))
    else:
        ap.print_help()
