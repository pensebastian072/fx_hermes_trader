"""Daily close panel resampled from the 1-minute archive, for B04+ (daily/monthly
horizon batteries).

Why a separate path from loader.build_panel: that function unions a MINUTE index
across every pair before it filters, which costs ~1.8GB for the full 28-pair
history and several times that in intermediates (see loader.build_panel's
docstring and the 2026-08-05 memory stall). A daily battery does not need any of
that. Here each pair is loaded, sanitized, resampled and released one at a time,
so peak memory is one pair (~8.4M rows) rather than the whole panel. Measured
2026-08-07: 9.8s/pair, ~4.6 min for 28 pairs, cached to parquet afterwards.

Rules this module is obliged to re-assert (CLAUDE.md gpu_meanrev rule 6): any
diagnostics/loading path that reimplements the loader must re-assert the holdout
clip, because duplicating build_panel duplicates away the one guard it exists to
enforce. `_clip_holdout` below is that re-assertion, on the same double gate
(unlock_holdout=True in code AND FX_MEANREV_HOLDOUT_UNLOCK=yes in env).

Sanitization is NOT reimplemented — this calls loader.load_pair_closes, so the
negative-price and wrong-scale filters (f877bec) apply to the daily panel too.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gpu_meanrev import config
from gpu_meanrev.data.loader import load_pair_closes

CACHE = config.DATA / "features" / "daily_closes.parquet"
CACHE_REPORT = config.DATA_REPORTS / "daily_closes_report.json"


def _clip_holdout(df: pd.DataFrame, unlock_holdout: bool) -> pd.DataFrame:
    if unlock_holdout and config.HOLDOUT_UNLOCK:
        return df
    return df[df.index < pd.Timestamp(config.HOLDOUT_START, tz="UTC")]


def _resample_one(pair: str, report: dict) -> pd.Series:
    """Sanitized 1-min closes -> one close per UTC calendar day, weekends dropped.

    Weekend handling: the FX week runs Sun 22:00 -> Fri 22:00 UTC, so a UTC-day
    bin makes Saturday empty and gives Sunday a ~2-hour stub. Both are dropped
    rather than kept, so no "day" in the panel is a 2-hour session masquerading
    as a full one. Monday's return therefore spans the weekend gap, which is the
    correct economics (you hold across the weekend) rather than an artifact.
    """
    s = load_pair_closes(pair, report=report)
    daily = s.resample("1D").last().dropna()
    daily = daily[daily.index.dayofweek < 5]
    daily.name = pair
    return daily


def build_daily_closes(pairs: list[str], unlock_holdout: bool = False) -> tuple[pd.DataFrame, dict]:
    """(n_days, n_pairs) daily close panel + the per-pair sanitize report.

    One pair in memory at a time. NaN where a pair had no print on a day that
    another pair traded — left NaN, never carried forward: a stale rate is a
    fake signal, and at this horizon there is no thin-liquidity-minute excuse
    for filling it.
    """
    report: dict = {}
    cols = []
    for pair in pairs:
        cols.append(_resample_one(pair, report))
    df = pd.concat(cols, axis=1, sort=True).sort_index()
    df = _clip_holdout(df, unlock_holdout)
    return df, report


def load_daily_closes(pairs: list[str], unlock_holdout: bool = False,
                      refresh: bool = False) -> pd.DataFrame:
    """Cached wrapper. The cache holds the FULL (pre-clip) panel; the clip is
    re-applied on every read, so a cache written under an unlocked holdout can
    never leak the holdout back to a locked caller."""
    cached: pd.DataFrame | None = None
    if CACHE.exists() and not refresh:
        cached = pd.read_parquet(CACHE)
        missing = [p for p in pairs if p not in cached.columns]
        if not missing:
            return _clip_holdout(cached[pairs], unlock_holdout)

    report: dict = {}
    cols = [_resample_one(p, report) for p in pairs]
    full = pd.concat(cols, axis=1, sort=True).sort_index()
    if cached is not None:
        # Keep columns a previous run already paid for. Writing back only the
        # pairs of THIS call would silently evict them and make the next caller
        # re-resample ~10s/pair for data that was already on disk.
        keep = [c for c in cached.columns if c not in full.columns]
        if keep:
            full = pd.concat([full, cached[keep]], axis=1, sort=True).sort_index()

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(CACHE)
    CACHE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    CACHE_REPORT.write_text(json.dumps({
        "pairs": pairs,
        "rows": len(full),
        "start": str(full.index.min()),
        "end": str(full.index.max()),
        "sanitize": report,
    }, indent=2, default=str), encoding="utf-8")

    return _clip_holdout(full, unlock_holdout)


def main() -> None:  # pragma: no cover - operator entry point
    import yaml
    pairs = yaml.safe_load(Path(config.RESOLVED_PAIRS_FILE).read_text(encoding="utf-8"))["pairs"]
    df = load_daily_closes(pairs, refresh=True)
    print(f"{len(df):,} days x {df.shape[1]} pairs  {df.index.min().date()} -> {df.index.max().date()}")
    print(f"cache: {CACHE}")
    print(f"NaN cells: {int(df.isna().sum().sum()):,} / {df.size:,}")


if __name__ == "__main__":  # pragma: no cover
    main()
