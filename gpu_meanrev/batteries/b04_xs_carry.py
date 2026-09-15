"""B04 — cross-sectional FX carry, monthly/quarterly rebalance.

WHY THIS BATTERY EXISTS
-----------------------
B01/B02/B03 all failed against the same wall: a gross edge of 0.406 bps per
round trip against an assumed 1.0 bp round-trip cost. The cost assumption was
2.5x the entire signal. No amount of extra minute-bar signal engineering moves
that, because the binding constraint is turnover, not predictive power.

B04 attacks the constraint instead of the signal. It changes three things at
once, deliberately:
  * effect class  -- a risk premium (carry), not a microstructure reversion
  * horizon       -- monthly/quarterly holds, not minutes
  * construction  -- cross-sectional rank across 8 currencies, not a per-pair
                     time series in isolation

At a monthly rebalance the strategy trades roughly 12 times a year instead of
thousands, so the cost hurdle that killed B01-B03 is ~2-3 orders of magnitude
smaller relative to the holding-period return.

HYPOTHESIS (pre-registered)
---------------------------
Sorting the 8 major currencies by their lagged 3-month interbank rate and
holding a long basket of the top k against a short basket of the bottom k earns
a positive risk-adjusted return, in-sample 2008-2022, at a turnover where
transaction cost is not the binding constraint.

PRIMARY PRE-COMMITTED TEST (kill criterion)
-------------------------------------------
At least one of the 8 registered combos must satisfy BOTH, on TOTAL return
(spot + carry accrual), in-sample:
    (a) gross annualised Sharpe >= 0.30, AND
    (b) breakeven cost >= 2.0x the assumed round-trip cost.
If no combo satisfies both: B04 is a FAIL. Stop. Do not expand the grid, do not
narrow the universe, do not spend the holdout.

WHAT A PASS WOULD AND WOULD NOT ESTABLISH
-----------------------------------------
Carry is one of the most heavily published premia in finance. Finding it here is
REPLICATION, not discovery, and the honest reading of a pass is "this box can
reproduce a known effect on this archive" -- not "we found an edge". Three
caveats are reported alongside every result rather than buried:

  1. Effective breadth is 8 currencies, not 28 pairs. The 28 pairs are the
     8-choose-2 combinations of the same 8 currencies, so they carry roughly 7
     independent bets, not 28. Pooled per-pair statistics would overstate n by
     ~4x -- the same clustering trap that inflated the options_desk sample.
  2. Carry pays small and steady and then unwinds violently (2008 and 2015 are
     both inside this window). A Sharpe computed over a period containing a
     crash is not the same statistic as one computed over a period that got
     lucky on timing, so a per-year table is reported, never just the pooled
     number.
  3. The carry leg assumes the INTERBANK differential. A retail account earns
     broker swap, which is materially worse and is not modelled here. The
     spot-only column is reported precisely because it is the part that does
     not depend on that assumption.

MULTIPLICITY
------------
Registered as a new family (`fx_xs_carry`) because the data, horizon and effect
class all differ from `fx_1min_meanrev`. That is defensible, but a new family
resets the trial count, which is also the classic way to launder multiplicity.
So the gate is reported TWICE: at the mechanical n_trials=8, and at a
conservative n_trials=288 that charges B04 for every prior FX trial on this box
(B01 160 + B02 72 + B03 48 + B04 8). If it only clears the mechanical count,
that is a weak result and is to be reported as one.

HOLDOUT
-------
Not spent. This run is in-sample only; loader/daily.py re-asserts the clip at
config.HOLDOUT_START (2023-01-01). B03's holdout is also still unspent and this
battery does not touch it.

COMMANDS
--------
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b04_xs_carry --register-only
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b04_xs_carry
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gpu_meanrev import config, gate
from gpu_meanrev.data import rates as rates_mod
from gpu_meanrev.data.daily import load_daily_closes
from gpu_meanrev.experiments import registry

BATTERY_ID = "B04_xs_carry"
FAMILY = "fx_xs_carry"

# Prior FX trials on this box, for the conservative multiplicity reading (see
# MULTIPLICITY above). B01's PILOT reused B01's grid and is not double-counted.
PRIOR_FX_TRIALS = 160 + 72 + 48

# Registered grid: 2 x 2 x 2 = 8 combos. Deliberately small -- every combo is a
# trial charged against the Deflated Sharpe, and this battery is testing one
# clean idea, not searching a space.
GRID = {
    "k": [2, 3],                          # basket size per side
    "rebalance": ["monthly", "quarterly"],
    "weighting": ["equal", "inverse_vol"],
}

VOL_LOOKBACK_DAYS = 60   # trailing window for inverse-vol weights, strictly causal
TRADING_DAYS = 252       # annualisation convention
IN_SAMPLE_START = "2008-01-01"   # common coverage start across all 28 pairs

KILL_MIN_SHARPE = 0.30
KILL_MIN_BREAKEVEN_MULT = 2.0


@dataclass(frozen=True)
class Combo:
    k: int
    rebalance: str
    weighting: str


def combos() -> list[Combo]:
    return [Combo(*v) for v in itertools.product(GRID["k"], GRID["rebalance"], GRID["weighting"])]


def n_combos() -> int:
    return len(combos())


# ── universe plumbing ───────────────────────────────────────────

def resolved_pairs() -> list[str]:
    return yaml.safe_load(Path(config.RESOLVED_PAIRS_FILE).read_text(encoding="utf-8"))["pairs"]


def pair_map(pairs: list[str]) -> dict[tuple[str, str], tuple[str, bool]]:
    """(base, quote) -> (pair symbol, inverted?) for every ordered currency pair.

    The 28-pair universe is exactly the 8-choose-2 combinations, so every leg is
    directly quotable and no synthetic cross has to be constructed. Fails loud if
    that stops being true rather than silently dropping a leg -- a missing leg
    would quietly bias the basket toward whichever currencies happen to be fully
    quoted.
    """
    have = set(pairs)
    out: dict[tuple[str, str], tuple[str, bool]] = {}
    for a, b in itertools.permutations(rates_mod.CURRENCIES, 2):
        if a + b in have:
            out[(a, b)] = (a + b, False)
        elif b + a in have:
            out[(a, b)] = (b + a, True)
        else:
            raise RuntimeError(f"no quote for {a}/{b} in the resolved universe")
    return out


def leg_returns(closes: pd.DataFrame, pmap: dict) -> dict[tuple[str, str], pd.Series]:
    """Daily simple return of being long `a` and short `b`, for every ordered (a, b).

    The inverted case uses the exact reciprocal return p[t-1]/p[t] - 1, not the
    negation of the direct return: -(p[t]/p[t-1] - 1) is only a first-order
    approximation and its error is systematically signed, which over 3,900 days
    would accumulate into the result.
    """
    out = {}
    for (a, b), (pair, inverted) in pmap.items():
        p = closes[pair]
        out[(a, b)] = (p.shift(1) / p - 1.0) if inverted else (p / p.shift(1) - 1.0)
    return out


# ── portfolio construction ──────────────────────────────────────

def rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> list[pd.Timestamp]:
    """Last available trading day of each month (or quarter) in the panel."""
    freq = "ME" if rebalance == "monthly" else "QE"
    s = pd.Series(index, index=index)
    return list(s.resample(freq).last().dropna())


def select_baskets(rate_row: pd.Series, k: int) -> tuple[list[str], list[str]]:
    """Top-k / bottom-k currencies by lagged short rate.

    Ties are broken by the currency's fixed order in rates_mod.CURRENCIES rather
    than by anything data-dependent, so a tie can never be resolved in the
    direction that happens to have performed better.
    """
    ranked = sorted(rate_row.dropna().items(), key=lambda kv: (-kv[1], rates_mod.CURRENCIES.index(kv[0])))
    if len(ranked) < 2 * k:
        return [], []
    longs = [c for c, _ in ranked[:k]]
    shorts = [c for c, _ in ranked[-k:]]
    return longs, shorts


def leg_weights(longs: list[str], shorts: list[str], weighting: str,
                legret: dict, asof: pd.Timestamp) -> dict[tuple[str, str], float]:
    """Weight per (long, short) leg, summing to 1.0 across the k*k legs.

    inverse_vol uses the trailing VOL_LOOKBACK_DAYS of each leg's return strictly
    BEFORE `asof` (`.loc[:asof]` then `.iloc[:-1]`), so the weight applied on the
    rebalance date never sees that date's own move.
    """
    legs = [(l, s) for l in longs for s in shorts]
    if not legs:
        return {}
    if weighting == "equal":
        w = {leg: 1.0 / len(legs) for leg in legs}
        return w

    inv = {}
    for leg in legs:
        hist = legret[leg].loc[:asof]
        hist = hist.iloc[:-1].tail(VOL_LOOKBACK_DAYS).dropna()
        sd = float(hist.std(ddof=1)) if len(hist) >= 20 else np.nan
        inv[leg] = (1.0 / sd) if (np.isfinite(sd) and sd > 0) else np.nan
    if not np.isfinite(list(inv.values())).any():
        return {leg: 1.0 / len(legs) for leg in legs}
    total = np.nansum(list(inv.values()))
    return {leg: (0.0 if not np.isfinite(v) else v / total) for leg, v in inv.items()}


def carry_accrual(longs_rate: float, shorts_rate: float, day_count: float) -> float:
    """ACT/365 accrual of the interbank differential, in return units."""
    return (longs_rate - shorts_rate) / 100.0 * day_count / 365.0


# ── one combo ───────────────────────────────────────────────────

def run_combo(combo: Combo, closes: pd.DataFrame, legret: dict, pmap: dict,
              sig_rates: pd.DataFrame) -> dict:
    index = closes.index
    rebals = rebalance_dates(index, combo.rebalance)
    rebal_set = set(rebals)

    weights: dict[tuple[str, str], float] = {}
    cur_rate_gap: dict[tuple[str, str], tuple[float, float]] = {}

    rows = []
    prev_date = None
    for date in index:
        # ORDER MATTERS. The existing book is marked over (prev_date, date] FIRST,
        # and only then is the portfolio rebalanced at the close of `date`. The
        # earlier arrangement rebalanced first and let the new weights earn the
        # rebalance day's own move: unbiased in expectation, but it silently made
        # the code a one-day-earlier strategy than the registration describes,
        # and misattributed one day of carry to the incoming basket.
        spot = 0.0
        carry = 0.0
        if weights and prev_date is not None:
            day_count = (date - prev_date).days
            for leg, w in weights.items():
                r = legret[leg].get(date, np.nan)
                if np.isfinite(r):
                    spot += w * r
                rl, rs = cur_rate_gap[leg]
                carry += w * carry_accrual(rl, rs, day_count)

        cost = 0.0
        if date in rebal_set:
            month = pd.Timestamp(date.year, date.month, 1)
            if month in sig_rates.index:
                row = sig_rates.loc[month]
                longs, shorts = select_baskets(row, combo.k)
                if longs and shorts:
                    new_w = leg_weights(longs, shorts, combo.weighting, legret, date)
                    for leg in set(new_w) | set(weights):
                        dw = abs(new_w.get(leg, 0.0) - weights.get(leg, 0.0))
                        if dw > 0:
                            pair, _ = pmap[leg]
                            cost += dw * config.cost_per_side(pair)
                    weights = new_w
                    cur_rate_gap = {
                        (l, s): (float(row[l]), float(row[s])) for l in longs for s in shorts
                    }

        if prev_date is not None:
            rows.append({"date": date, "spot": spot, "carry": carry, "cost": cost})
        prev_date = date

    df = pd.DataFrame(rows).set_index("date")
    df["gross_total"] = df["spot"] + df["carry"]
    df["net_total"] = df["gross_total"] - df["cost"]
    df["net_spot"] = df["spot"] - df["cost"]
    return {"combo": asdict(combo), "daily": df}


# ── metrics ─────────────────────────────────────────────────────

def _ann_sharpe(x: pd.Series) -> float:
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(x.mean() / sd * np.sqrt(TRADING_DAYS))


def summarise(res: dict) -> dict:
    df = res["daily"]
    gross_sum = float(df["gross_total"].sum())
    cost_sum = float(df["cost"].sum())
    breakeven_mult = (gross_sum / cost_sum) if cost_sum > 0 else float("inf")

    assumed_bps = float(np.mean([config.COST_PER_SIDE["major"], config.COST_PER_SIDE["cross"]])) * 1e4
    n_years = len(df) / TRADING_DAYS

    out = {
        "combo": res["combo"],
        "n_days": int(len(df)),
        "years": round(n_years, 2),
        "ann_return_gross_total": round(gross_sum / n_years, 5),
        "ann_return_net_total": round(float(df["net_total"].sum()) / n_years, 5),
        "ann_return_net_spot": round(float(df["net_spot"].sum()) / n_years, 5),
        "ann_sharpe_gross_total": round(_ann_sharpe(df["gross_total"]), 4),
        "ann_sharpe_net_total": round(_ann_sharpe(df["net_total"]), 4),
        "ann_sharpe_net_spot": round(_ann_sharpe(df["net_spot"]), 4),
        # Rule 1 pairing: breakeven cost is reported NEXT TO profit factor, not
        # only inside the gate dict, so a scorecard read on its own still shows
        # the comparison that B01/B02 lacked for two days.
        "profit_factor_net_total": round(gate.profit_factor(df["net_total"].to_numpy()), 4),
        "profit_factor_gross_total": round(gate.profit_factor(df["gross_total"].to_numpy()), 4),
        "n_rebalances": int((df["cost"] > 0).sum()),
        "total_cost_paid": round(cost_sum, 6),
        "breakeven_cost_multiple": round(breakeven_mult, 2) if np.isfinite(breakeven_mult) else None,
        "assumed_cost_per_side_bps": round(assumed_bps, 3),
        "breakeven_cost_per_side_bps": (
            round(assumed_bps * breakeven_mult, 3) if np.isfinite(breakeven_mult) else None
        ),
    }
    out["kill_pass"] = bool(
        np.isfinite(out["ann_sharpe_gross_total"])
        and out["ann_sharpe_gross_total"] >= KILL_MIN_SHARPE
        and breakeven_mult >= KILL_MIN_BREAKEVEN_MULT
    )
    return out


def per_year_table(df: pd.DataFrame) -> list[dict]:
    """Per-year rows. Reported ALWAYS, never replaced by the pooled number --
    carry's whole failure mode is a long calm stretch punctuated by a crash, and
    a pooled Sharpe hides exactly that."""
    rows = []
    for year, g in df.groupby(df.index.year):
        rows.append({
            "year": int(year),
            "n_days": int(len(g)),
            "ret_net_total": round(float(g["net_total"].sum()), 5),
            "ret_net_spot": round(float(g["net_spot"].sum()), 5),
            "ann_sharpe_net_total": round(_ann_sharpe(g["net_total"]), 3),
        })
    return rows


# ── entry points ────────────────────────────────────────────────

def register_only() -> dict:
    hypothesis = (
        "Cross-sectional FX carry: sorting the 8 major currencies by lagged 3-month "
        "interbank rate (FRED OECD IR3TIB01, publication-lagged 1 month) and holding "
        "the top-k long against the bottom-k short earns a positive risk-adjusted "
        "return in-sample 2008-2022 at a turnover where transaction cost is not "
        "binding -- the constraint that killed B01/B02/B03, where a 0.406 bps gross "
        "edge faced a 1.0 bp round-trip cost. PRIMARY PRE-COMMITTED TEST: at least "
        "one of the 8 registered combos reaches gross annualised Sharpe >= 0.30 AND "
        "breakeven cost >= 2.0x assumed round-trip cost, on total return (spot + "
        "carry accrual). If not, B04 is a FAIL: stop, do not expand the grid, do not "
        "narrow the universe, do not spend the holdout. Reported with a per-year "
        "table (carry crashes rather than decays), a spot-only column (the interbank "
        "differential is not what a retail account earns), an effective-breadth note "
        "(8 currencies, ~7 independent bets, NOT 28 pairs), and the gate evaluated "
        "at both n_trials=8 and a conservative n_trials=288 charging every prior FX "
        "trial on this box. Carry is a heavily published premium: a pass is "
        "replication, not discovery."
    )
    try:
        path = registry.register(
            battery_id=BATTERY_ID,
            hypothesis=hypothesis,
            dataset_family=FAMILY,
            pairs=resolved_pairs(),
            param_grid={
                **GRID,
                "vol_lookback_days": [VOL_LOOKBACK_DAYS],
                "publication_lag_months": [rates_mod.PUBLICATION_LAG_MONTHS],
                "rate_series": [rates_mod.SERIES],
                "in_sample": [f"{IN_SAMPLE_START} .. {config.HOLDOUT_START} (exclusive)"],
                "kill_criterion": [{
                    "metric_a": "ann_sharpe_gross_total",
                    "threshold_a": KILL_MIN_SHARPE,
                    "metric_b": "breakeven_cost_multiple",
                    "threshold_b": KILL_MIN_BREAKEVEN_MULT,
                    "combine": "at least one combo must satisfy BOTH",
                    "on_failure": "report FAIL, stop, no grid expansion, no holdout spend",
                }],
            },
            n_trials=n_combos(),
        )
        return {"registered": str(path), "n_trials": n_combos()}
    except FileExistsError:
        return {"registered": "already", "n_trials": n_combos()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--register-only", action="store_true")
    args = ap.parse_args(argv)

    if args.register_only:
        print(json.dumps(register_only(), indent=2, default=str))
        return

    reg = registry.load_registration(BATTERY_ID)  # refuses to run unregistered
    pairs = reg["pairs"]

    closes = load_daily_closes(pairs)
    closes = closes[closes.index >= pd.Timestamp(IN_SAMPLE_START, tz="UTC")]
    sig = rates_mod.signal_rates(
        start=str(closes.index.min().date()), end=str(closes.index.max().date())
    )

    pmap = pair_map(pairs)
    legret = leg_returns(closes, pmap)

    summaries = []
    per_year = {}
    daily_by_combo = {}
    for combo in combos():
        res = run_combo(combo, closes, legret, pmap, sig)
        s = summarise(res)
        summaries.append(s)
        key = f"k{combo.k}_{combo.rebalance}_{combo.weighting}"
        per_year[key] = per_year_table(res["daily"])
        daily_by_combo[key] = res["daily"]["net_total"]
        registry.log_trial(BATTERY_ID, FAMILY, asdict(combo), pairs)

    passing = [s for s in summaries if s["kill_pass"]]
    best = max(summaries, key=lambda s: (s["ann_sharpe_gross_total"]
                                         if np.isfinite(s["ann_sharpe_gross_total"]) else -9e9))
    best_key = "k{k}_{rebalance}_{weighting}".format(**best["combo"])
    best_daily = daily_by_combo[best_key].to_numpy()

    payload = {
        "battery_id": BATTERY_ID,
        "family": FAMILY,
        "window": {"start": str(closes.index.min().date()), "end": str(closes.index.max().date())},
        "holdout_spent": False,
        "n_combos": n_combos(),
        "summaries": summaries,
        "kill_criterion": {
            "min_ann_sharpe_gross": KILL_MIN_SHARPE,
            "min_breakeven_multiple": KILL_MIN_BREAKEVEN_MULT,
            "n_passing": len(passing),
            "verdict": "PASS" if passing else "FAIL",
        },
        "best_combo": best,
        "per_year": per_year,
        "effective_breadth": {
            "n_pairs_traded": len(pairs),
            "n_currencies": len(rates_mod.CURRENCIES),
            "independent_bets": len(rates_mod.CURRENCIES) - 1,
            "note": "28 pairs are the 8-choose-2 combinations of 8 currencies; "
                    "per-pair n overstates the sample by roughly 4x",
        },
        "reporting_notes": [
            "`best_combo` is chosen on GROSS Sharpe but the gate is fed the NET "
            "daily series — selecting on gross and grading on net is the "
            "conservative direction, but the two numbers are not the same metric.",
            "`assumed_cost_per_side_bps` is the unweighted mean of the major/cross "
            "tiers, so `breakeven_cost_per_side_bps` is an approximation; "
            "`breakeven_cost_multiple` is exact.",
            "Positions are marked over (prev_date, date] and rebalanced at the "
            "close of `date`; inverse-vol weights see returns through date-1 only.",
        ],
        "gate_mechanical": gate.evaluate_gate(best_daily, n_trials=n_combos()),
        "gate_conservative": gate.evaluate_gate(
            best_daily, n_trials=n_combos() + PRIOR_FX_TRIALS
        ),
    }

    out = registry.write_scorecard(BATTERY_ID, payload)
    verdict = payload["gate_conservative"]
    registry.append_result_row(
        BATTERY_ID, verdict,
        note=f"kill={payload['kill_criterion']['verdict']} best={best_key} "
             f"breakeven={best['breakeven_cost_multiple']}x holdout unspent",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "per_year"},
                     indent=2, default=str))
    print(f"\nscorecard: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
