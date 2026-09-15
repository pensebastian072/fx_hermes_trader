"""Cached parquet -> common UTC minute grid across the resolved pair set.

Builds a union-of-timestamps master index across all requested pairs, and
reindexes each pair's close price onto it. Gaps up to
config.MAX_FORWARD_FILL_MINUTES are forward-filled (thin-liquidity minutes
with no tick); anything longer is left NaN rather than faked, so a genuine
dead spot can't manufacture a fake reversion signal or fake cross-pair
correlation (see config.py for the real gap-density numbers this threshold
was set from).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gpu_meanrev import config


def _ticks_path(pair: str) -> Path:
    return config.RAW / pair.lower() / "ticks.parquet"


# A tick this far from its own rolling median is a corrupt print, not a market
# move. Deliberately loose: the 2015 CHF unpeg was ~-29% in minutes and MUST
# survive this filter, while the real defects found in this archive were 2-3
# ORDERS OF MAGNITUDE off (AUDJPY printing 0.67 against a ~85 median).
OUTLIER_FACTOR = 4.0
_MEDIAN_WINDOW = 61  # minutes, centred

# Band around the WHOLE-SERIES median. The local (centred) filter above only
# catches ISOLATED bad prints: a wrong-scale run longer than the window drags
# the local median with it, so the ratio looks normal and the whole segment
# survives. That is not hypothetical -- AUDJPY 2005-05-01 onward has 7,602
# contiguous minutes printing 0.67-2.28 against an ~85 median, and the local
# filter removed none of them.
# 4x is safe for this universe over 2000-2025: the widest real excursion is
# USDJPY ~76-160 against a ~110 median (0.69x-1.45x), nowhere near the band.
GLOBAL_BAND_FACTOR = 4.0


def sanitize_closes(s: pd.Series) -> tuple[pd.Series, dict]:
    """Drop physically impossible and grossly off-scale prints.

    Real defects found in elthariel/histdata_fx_1m (2026-08-05), both of which
    silently produced fake profit:
      * EURUSD 2001-09-12 01:12 close = -0.0001 (a NEGATIVE FX price). Divided
        into the return formula it yielded a single trade PnL of 9.1e11.
      * AUDJPY 2005 prints near 0.67 against a ~85 median (decimal / wrong
        series). Entering at 0.67 and exiting at 85 books a 126x "return";
        347 such trades inflated 2005's profit factor to 111.6 while the
        median trade was +0.0005.

    Neither is caught by a NaN check (they are finite numbers), which is
    exactly why they survived to the scorecard. Returns the cleaned series
    plus a report of what was removed — callers should surface a non-zero
    report rather than swallow it.
    """
    n0 = len(s)
    # NaN must be counted separately: `s[s > 0]` silently discards it (NaN > 0
    # is False), which made an early version of this report claim "2 dropped"
    # while actually removing 20,315 rows from EURUSD. A drop report that does
    # not reconcile to rows_in - rows_out is worse than no report.
    n_nan = int(s.isna().sum())
    nonpositive = int((s <= 0).sum())
    s = s[s > 0]

    # Stage 1 -- GLOBAL band. Catches sustained wrong-scale segments that the
    # local filter structurally cannot see (see GLOBAL_BAND_FACTOR).
    global_med = s.median()
    out_of_band_mask = (s > global_med * GLOBAL_BAND_FACTOR) | (s < global_med / GLOBAL_BAND_FACTOR)
    out_of_band = int(out_of_band_mask.sum())
    s = s[~out_of_band_mask]

    # Stage 2 -- LOCAL centred median. Catches isolated spikes that sit inside
    # the global band. NOTE: this is a look-ahead-based FILTER (it consults +/-30
    # minutes around each bar). It only ever REMOVES bars, never creates or
    # shifts a signal, so it cannot manufacture an edge -- but it is not causal
    # and should not be reused as a feature.
    med = s.rolling(_MEDIAN_WINDOW, center=True, min_periods=5).median()
    med = med.bfill().ffill()
    ratio = s / med
    off_scale_mask = (ratio > OUTLIER_FACTOR) | (ratio < 1.0 / OUTLIER_FACTOR)
    off_scale = int(off_scale_mask.sum())
    s = s[~off_scale_mask]

    rep = {
        "rows_in": n0, "rows_out": len(s),
        "dropped_nan": n_nan,
        "dropped_nonpositive": nonpositive,
        "dropped_out_of_band": out_of_band,
        "dropped_off_scale": off_scale,
    }
    accounted = n_nan + nonpositive + out_of_band + off_scale
    rep["unaccounted"] = n0 - len(s) - accounted
    assert rep["unaccounted"] == 0, f"drop accounting does not reconcile: {rep}"
    return s, rep


def load_pair_closes(pair: str, report: dict | None = None) -> pd.Series:
    """Deduped, sorted, UTC-indexed, SANITIZED close price series for one pair."""
    path = _ticks_path(pair)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run gpu_meanrev.data.hf_download first")
    df = pd.read_parquet(path, columns=["ts", "close"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # ~0.004% duplicate timestamps observed in the probe (300-360 / 8.4M rows)
    # — keep the last tick for a given minute, drop the rest.
    df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts")
    s = df.set_index("ts")["close"]
    s, rep = sanitize_closes(s)
    if report is not None:
        report[pair] = rep
    s.name = pair
    return s


def build_panel(pairs: list[str], unlock_holdout: bool = False,
                start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Union UTC minute index across `pairs`, gap-aware forward-fill.

    Returns a (n_bars, n_pairs) DataFrame of close prices, columns=pairs.
    A cell is NaN if that pair had no observation within
    MAX_FORWARD_FILL_MINUTES of that timestamp — never silently carried
    forward across a real dead spot.

    Clips to bars < config.HOLDOUT_START unless BOTH unlock_holdout=True
    here AND FX_MEANREV_HOLDOUT_UNLOCK=yes are set (config.HOLDOUT_UNLOCK) —
    same double-gate as alpaca_gpu_lab, meant to be flipped once, at the end.

    start_date/end_date, if given, filter each pair's series BEFORE the
    union/ffill steps (not just the final result) — those steps are
    O(full history) in both time and MEMORY, and dominate both: the full
    2000-2025 28-pair panel is ~1.8GB for the result alone and several times
    that in intermediates, which on this 16GB box means swap-thrashing.
    A run over a narrow window must not pay that cost for data it is about
    to discard. (Learned the hard way 2026-08-05: two concurrent jobs each
    building a full panel drove free memory to 0.9GB and stalled both.)
    """
    series = {p: load_pair_closes(p) for p in pairs}
    if start_date is not None:
        cutoff = pd.Timestamp(start_date, tz="UTC")
        series = {p: s[s.index >= cutoff] for p, s in series.items()}
    if end_date is not None:
        cutoff_end = pd.Timestamp(end_date, tz="UTC")
        series = {p: s[s.index < cutoff_end] for p, s in series.items()}
    union_index = series[pairs[0]].index
    for p in pairs[1:]:
        union_index = union_index.union(series[p].index)
    union_index = union_index.sort_values()

    limit = config.MAX_FORWARD_FILL_MINUTES
    panel = {}
    for p, s in series.items():
        reindexed = s.reindex(union_index)
        panel[p] = reindexed.ffill(limit=limit)
    df = pd.DataFrame(panel, index=union_index)

    if not (unlock_holdout and config.HOLDOUT_UNLOCK):
        df = df[df.index < pd.Timestamp(config.HOLDOUT_START, tz="UTC")]
    return df


def common_date_range(pairs: list[str]) -> dict:
    """Per-pair min/max timestamp + the intersection every pair actually covers.

    Used to set config.HOLDOUT_START honestly (can't assume alpaca_gpu_lab's
    2026-01-01 boundary — this archive's real end date must be measured).
    """
    ranges = {}
    for p in pairs:
        s = load_pair_closes(p)
        ranges[p] = {"min": s.index.min(), "max": s.index.max(), "n": len(s)}
    common_start = max(r["min"] for r in ranges.values())
    common_end = min(r["max"] for r in ranges.values())
    return {"per_pair": ranges, "common_start": common_start, "common_end": common_end}
