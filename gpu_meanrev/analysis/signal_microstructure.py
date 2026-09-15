"""Is the 0.4bp gross edge a real reversion, or a stale-price artifact?

Motivation. `data/loader.build_panel` forward-fills a pair's close across
gaps up to MAX_FORWARD_FILL_MINUTES (10). A forward-filled run is a FLAT
segment: it depresses the rolling std, so when the next real tick lands the
z-score of that tick is mechanically inflated, and the "reversion" that
follows is partly just the price resuming its true path after a hole in the
data. This box has been burned by exactly this shape before — the
research_ledger lead-lag "signal" that turned out to be 100% INDA stale-NAV.
B01 and B02 both entered on |z| >= entry_z with no staleness condition at
all, so neither could tell the two apart.

Approach. Skip the state machine entirely and measure the SIGNAL, not the
strategy: at every bar where |z| >= entry_z, take the signed forward return
over a set of horizons, and split those events by how much of the lookback
window was forward-filled rather than genuinely traded. If the edge survives
on windows with zero fill, it is a real (if tiny) reversion. If it lives in
the contaminated bucket, B01/B02's gross edge was never tradable at any cost.

Two things fall out for free, neither of which the batteries measured:
  * the reversion HALF-LIFE (edge vs horizon), which is what actually sets a
    sensible exit rule — B01/B02 fixed max_hold = 4x lookback by assertion;
  * a bar-level staleness and realised-volatility profile per year, to test
    whether the 2015->2022 gross-edge decay tracks falling volatility (a real
    regime change) or falling data-gap density (an artifact of the archive
    getting denser over time).

Nothing here selects a strategy or charges a trial: it is measurement of a
closed battery's input data, reported per year, holdout untouched.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.analysis.signal_microstructure --years 2022
  .venv\\Scripts\\python.exe -m gpu_meanrev.analysis.signal_microstructure --full
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from gpu_meanrev import config
from gpu_meanrev.batteries.b02_mtf_confluence import MAJORS_7
from gpu_meanrev.data.loader import load_pair_closes
from gpu_meanrev.features.rolling import rolling_zscore

OUT_PATH = config.SCORECARDS / "B02_signal_microstructure.json"

# Combo 64's parameters — the battery's own headline, reused as-is so this is
# a diagnosis of that result and not a fresh search over lookbacks.
LOOKBACK = 240
ENTRY_Z = 2.0

# Forward horizons in minutes. 960 = combo 64's max_hold (4 x lookback).
HORIZONS = [1, 5, 15, 30, 60, 120, 240, 480, 960]

# Fraction of the lookback window that was forward-filled rather than traded.
# NOT a partition: the (0.0, 0.02) bucket has an inclusive lower bound, so it
# CONTAINS every zero-fill event rather than sitting beside it. The (0.02, .]
# and (0.10, .] buckets are exclusive and so are genuinely disjoint. Labelled
# in the output so nobody reads the four as summing to the whole.
STALE_BUCKETS = [(0.0, 0.0), (0.0, 0.02), (0.02, 0.10), (0.10, 1.01)]


def build_panel_with_fill_mask(pairs: list[str], start: str, end: str):
    """Same union/ffill panel `build_panel` makes, plus the mask of which
    cells are forward-filled rather than observed.

    Deliberately duplicates loader.build_panel's steps instead of calling it:
    the mask has to be captured BETWEEN the reindex and the ffill, and the
    loader returns only the filled frame. Kept adjacent to it so a change
    there is visible here.

    Duplicating the loader also duplicates AWAY its holdout clip, which is the
    one thing that function exists to enforce — so the clip is re-asserted
    here rather than left to every caller. Diagnostics never need 2023+; if
    one ever does, it goes through the loader's real double gate
    (unlock_holdout + FX_MEANREV_HOLDOUT_UNLOCK), not through this shortcut.
    """
    if pd.Timestamp(end, tz="UTC") > pd.Timestamp(config.HOLDOUT_START, tz="UTC"):
        raise ValueError(
            f"end={end} reaches into the locked holdout ({config.HOLDOUT_START}); "
            "this diagnostics path has no unlock and must not acquire one"
        )
    series = {p: load_pair_closes(p) for p in pairs}
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    series = {p: s[(s.index >= lo) & (s.index < hi)] for p, s in series.items()}
    idx = series[pairs[0]].index
    for p in pairs[1:]:
        idx = idx.union(series[p].index)
    idx = idx.sort_values()

    filled, was_missing = {}, {}
    for p, s in series.items():
        r = s.reindex(idx)
        was_missing[p] = r.isna()
        filled[p] = r.ffill(limit=config.MAX_FORWARD_FILL_MINUTES)
    panel = pd.DataFrame(filled, index=idx)
    # A cell is FORWARD-FILLED iff it had no observation but ended up with a
    # value. Cells still NaN after the ffill were beyond the 10-min limit and
    # are excluded from everything downstream anyway.
    fill_mask = pd.DataFrame(was_missing, index=idx) & panel.notna()
    return panel, fill_mask


def _stale_fraction(fill_mask_row: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling mean of the fill flag over `window` bars, same causal
    alignment as rolling_zscore (index t = window ending at t)."""
    x = fill_mask_row.double().unsqueeze(1)
    m = torch.nn.functional.avg_pool1d(x, kernel_size=window, stride=1).squeeze(1)
    out = torch.full(fill_mask_row.shape, float("nan"), dtype=torch.float64)
    out[:, window - 1:] = m
    return out


def analyse_year(year: int, pairs: list[str]) -> dict:
    panel, fill_mask = build_panel_with_fill_mask(
        pairs, f"{year}-01-01", f"{year + 1}-01-01")
    if panel.empty:
        return {"year": year, "empty": True}

    px = torch.tensor(panel.to_numpy(dtype="float64").T)          # (n_pairs, n_bars)
    fm = torch.tensor(fill_mask.to_numpy().T)
    z = rolling_zscore(px.float(), LOOKBACK)
    stale = _stale_fraction(fm, LOOKBACK)

    valid_px = torch.isfinite(px) & (px > 0)
    # Signed exposure: fade the move. z >= +entry -> short (-1), z <= -entry -> long (+1).
    side = torch.zeros_like(z)
    side[z >= ENTRY_Z] = -1.0
    side[z <= -ENTRY_Z] = 1.0
    event = (side != 0) & valid_px & torch.isfinite(stale)

    n_bars = px.shape[1]
    res_h, res_bucket = [], []
    for h in HORIZONS:
        if h >= n_bars:
            continue
        fwd = torch.full_like(px, float("nan"))
        fwd[:, :n_bars - h] = (px[:, h:] - px[:, :n_bars - h]) / px[:, :n_bars - h]
        ok = event & torch.isfinite(fwd) & valid_px
        signed = (side * fwd)[ok]
        res_h.append({
            "horizon_min": h, "n_events": int(ok.sum()),
            "signed_bps": round(float(signed.mean()) * 1e4, 4),
            # NOT a usable significance statistic — see t_stat_warning below.
            "t_stat_iid_naive": round(float(signed.mean() / (signed.std(unbiased=True)
                                                            / np.sqrt(signed.numel()))), 3),
        })
        if h == 960:
            for lo, hi in STALE_BUCKETS:
                sel = ok & ((stale > lo) if lo > 0 else (stale >= lo)) & (stale <= hi)
                if lo == 0.0 and hi == 0.0:
                    sel = ok & (stale == 0.0)
                v = (side * fwd)[sel]
                res_bucket.append({
                    "label": ("fill == 0" if hi == 0.0 else
                              f"fill <= {hi:.0%} (INCLUDES fill==0)" if lo == 0.0 else
                              f"{lo:.0%} < fill <= {min(hi, 1.0):.0%}"),
                    "stale_frac_lo": lo, "stale_frac_hi": hi,
                    "disjoint_from_zero_bucket": lo > 0.0,
                    "n_events": int(v.numel()),
                    "signed_bps": round(float(v.mean()) * 1e4, 4) if v.numel() else None,
                })

    logret = torch.log(px[:, 1:] / px[:, :-1])
    lr = logret[torch.isfinite(logret)]
    return {
        "year": year,
        "n_bars": int(n_bars),
        "pct_cells_forward_filled": round(float(fm.double().mean()) * 100, 3),
        "pct_cells_nan_after_fill": round(float((~valid_px).double().mean()) * 100, 3),
        "realised_vol_1m_bps": round(float(lr.std(unbiased=True)) * 1e4, 4),
        "n_signal_events": int(event.sum()),
        "events_per_pair_day": round(float(event.sum()) / len(pairs) / 260, 1),
        "forward_return_by_horizon": res_h,
        "h960_by_stale_bucket": res_bucket,
    }


def delay_curve(year: int, pairs: list[str], delays=(0, 1, 2, 5, 10)) -> dict:
    """What does a non-zero execution latency cost the signal?

    B01 and B02 both compute z from a window ENDING at bar t and then enter at
    bar t's own close — a zero-latency assumption that was never stated as one
    or tested. Nothing in the archive can tell us the real fill, but the
    sensitivity is measurable: shift the entry (and the horizon with it) by d
    bars and watch the signed forward return. If a one-minute delay eats a
    large share of a 0.4bp edge, the study's headline number is an upper bound
    on something unreachable rather than an estimate of anything.
    """
    panel, _fm = build_panel_with_fill_mask(pairs, f"{year}-01-01", f"{year + 1}-01-01")
    px = torch.tensor(panel.to_numpy(dtype="float64").T)
    z = rolling_zscore(px.float(), LOOKBACK)
    valid = torch.isfinite(px) & (px > 0)
    side = torch.zeros_like(z)
    side[z >= ENTRY_Z] = -1.0
    side[z <= -ENTRY_Z] = 1.0
    n_bars = px.shape[1]
    h = 960

    rows = []
    for d in delays:
        # Entry at t+d, exit h bars after THAT — the trade is delayed, not shortened.
        end = n_bars - h - d
        if end <= 0:
            continue
        entry = px[:, d:d + end]
        exit_ = px[:, d + h:d + h + end]
        sd = side[:, :end]
        ok = (sd != 0) & valid[:, :end] & valid[:, d:d + end] & valid[:, d + h:d + h + end]
        r = (sd * (exit_ - entry) / entry)[ok]
        rows.append({"delay_min": d, "n_events": int(r.numel()),
                     "signed_bps": round(float(r.mean()) * 1e4, 4)})
    base = rows[0]["signed_bps"] if rows else None
    for r in rows:
        r["pct_of_zero_latency"] = round(100 * r["signed_bps"] / base, 1) if base else None
    return {"year": year, "horizon_min": h, "curve": rows}


def main(years: list[int]) -> dict:
    t0 = time.time()
    out = {
        "note": "signal-level diagnostics on B02's inputs; selects nothing, "
                "charges no trial, holdout untouched",
        "t_stat_warning": (
            "t_stat_iid_naive assumes independent events. It is NOT: ~390k "
            "events per year overlap by up to 960 minutes and run across 7 "
            "USD-correlated pairs, so the effective sample is smaller than n "
            "by roughly an order of magnitude and these t-stats are inflated "
            "by about the square root of that. Use them to compare shapes "
            "across horizons, never as evidence of significance — the honest "
            "significance number for this study is the day-aggregated t=1.12 "
            "in B02_mtf_confluence_diagnostics.json."
        ),
        "params": {"lookback_min": LOOKBACK, "entry_z": ENTRY_Z,
                   "pairs": MAJORS_7, "horizons_min": HORIZONS,
                   "params_note": "lookback/entry_z are combo 64's, i.e. B02's "
                                  "ex-post best combo. Fine for diagnosing that "
                                  "result; any forward claim needs a fresh "
                                  "pre-registration."},
        "per_year": [],
    }
    for y in years:
        r = analyse_year(y, MAJORS_7)
        out["per_year"].append(r)
        print(json.dumps(r, default=str))
    out["elapsed_sec"] = round(time.time() - t0, 1)

    prev = json.loads(OUT_PATH.read_text(encoding="utf-8")) if OUT_PATH.exists() else None
    if prev:  # merge so a partial run is resumable year by year
        done = {r["year"]: r for r in prev.get("per_year", [])}
        done.update({r["year"]: r for r in out["per_year"]})
        out["per_year"] = [done[k] for k in sorted(done)]
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--delays", action="store_true",
                    help="execution-latency sensitivity on the first and last fold year")
    a = ap.parse_args()
    if a.delays:
        out = {"note": "execution-latency sensitivity; B01/B02 both assume delay=0",
               "years": [delay_curve(y, MAJORS_7) for y in (a.years or [2015, 2022])]}
        path = config.SCORECARDS / "B02_execution_delay.json"
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {path}")
    else:
        main(list(range(2015, 2023)) if a.full else (a.years or [2022]))
