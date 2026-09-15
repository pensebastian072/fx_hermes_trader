"""Per-year x per-pair regime tables from a finished battery's trade log.

The box's standing rule (CLAUDE.md): prefer a within-year or per-fold table
to a pooled number that a sample skew could have manufactured. A pooled
15-year verdict hides the thing we actually want to know -- WHICH regimes
the effect lived in, and how the cross-pair relationships shifted year to
year (2015 CHF unpeg, 2020 covid, 2022 BoJ/yield divergence are not the
same market).

Reads the checkpointed per-trade records (which carry pair, ts, pnl,
combo_idx) -- so this costs no GPU and no re-run.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.reporting.regime_table B01_zscore_reversion
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from gpu_meanrev import config, gate


def _trade_files(battery_id: str) -> tuple[list, list]:
    ckpt_dir = config.DATA_REPORTS / f"{battery_id}_checkpoint" / "trades"
    block_parts = sorted(ckpt_dir.glob("block_*.parquet"))
    year_parts = sorted(ckpt_dir.glob("year_*.parquet"))
    if not block_parts and not year_parts:
        raise FileNotFoundError(f"no trade parquet files in {ckpt_dir}")
    return year_parts, block_parts


def best_combo_streaming(battery_id: str, min_trades: int = 8) -> int:
    """Pick the highest mean/std combo WITHOUT loading the whole log.

    B01's full 2008-2022 log is ~107M trade rows across 15 files. Reading it
    into one DataFrame — which this module originally did — needs tens of GB
    on a 16GB box and cannot complete. Only two columns are needed to rank
    combos, and Welford-style running sums over per-file reads make it a
    streaming problem instead of a memory one.

    Selecting the winner ex post is the pre-existing (and deliberate)
    behaviour: the deflation charges the FULL grid size via n_trials, which
    is what makes an ex-post pick honest rather than free.
    """
    year_parts, block_parts = _trade_files(battery_id)
    # Ranking must use the SAME de-duplicated row set the table is later built
    # from, or a checkpoint holding both layouts would rank on double-counted
    # years (and apply min_trades to an inflated count) and could pick a combo
    # that load_trades then reports on fewer rows. No battery has both layouts
    # today; this keeps the two paths in agreement if one ever does.
    years_covered = {int(p.stem.split("_")[1]) for p in year_parts}
    n = {}
    s1 = {}
    s2 = {}
    for p in list(year_parts) + list(block_parts):
        is_block = p.name.startswith("block_")
        cols = ["combo_idx", "pnl", "ts"] if is_block else ["combo_idx", "pnl"]
        df = pd.read_parquet(p, columns=cols)
        if df.empty:
            continue
        if is_block and years_covered:
            df = df[~pd.to_datetime(df["ts"], utc=True).dt.year.isin(years_covered)]
            if df.empty:
                continue
        g = df.groupby("combo_idx")["pnl"].agg(["count", "sum", lambda x: float((x ** 2).sum())])
        g.columns = ["count", "sum", "sumsq"]
        for idx, row in g.iterrows():
            n[idx] = n.get(idx, 0) + int(row["count"])
            s1[idx] = s1.get(idx, 0.0) + float(row["sum"])
            s2[idx] = s2.get(idx, 0.0) + float(row["sumsq"])
        del df, g

    best, best_score = None, -np.inf
    for idx, cnt in n.items():
        if cnt < min_trades:
            continue
        mean = s1[idx] / cnt
        var = max(s2[idx] / cnt - mean * mean, 0.0) * (cnt / (cnt - 1))
        if var <= 0:
            continue
        score = mean / np.sqrt(var)
        if score > best_score:
            best, best_score = int(idx), score
    if best is None:
        raise ValueError(f"no combo in {battery_id} had >= {min_trades} trades")
    return best


def load_trades(battery_id: str, combo_idx: int | None = None) -> pd.DataFrame:
    """Read a battery's trade log from either checkpoint layout, no overlap.

    Two runners have written trades for B01: the original whole-panel one
    (block_*.parquet, 90-day blocks that straddle year boundaries) and the
    memory-safe year-at-a-time one (year_<Y>.parquet). A year covered by a
    year-file is authoritative for that year; block-file rows are used only
    for years no year-file covers. Filtering on the timestamp (not the file)
    makes that exact even though blocks straddle year ends.

    `combo_idx` filters AT READ TIME, which is what makes the full 15-year
    log tractable: one combo out of a 160-combo grid is ~1/160th of 107M
    rows. Passing None restores the old load-everything behaviour and will
    exhaust memory on a full B01 log — it exists for small batteries only.

    Only the columns the regime tables actually use are read, which also
    sidesteps the mixed schema across vintages: files written before
    2026-08-06 have no entry-side columns, later ones do.
    """
    year_parts, block_parts = _trade_files(battery_id)
    cols = ["ts", "pair", "combo_idx", "pnl"]
    flt = [("combo_idx", "==", combo_idx)] if combo_idx is not None else None

    frames = []
    years_covered: set[int] = set()
    for p in year_parts:
        df = pd.read_parquet(p, columns=cols, filters=flt)
        # A year file is authoritative for its year even if this combo took
        # no trades in it — derive coverage from the FILENAME, not the rows,
        # or an empty filtered read would silently re-admit block rows.
        years_covered.add(int(p.stem.split("_")[1]))
        if df.empty:
            continue
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["year"] = df["ts"].dt.year
        frames.append(df)

    if block_parts:
        bframes = []
        for p in block_parts:
            bdf = pd.read_parquet(p, columns=cols, filters=flt)
            if bdf.empty:
                continue
            bdf["ts"] = pd.to_datetime(bdf["ts"], utc=True)
            bdf["year"] = bdf["ts"].dt.year
            bdf = bdf[~bdf["year"].isin(years_covered)]
            if not bdf.empty:
                bframes.append(bdf)
        frames.extend(bframes)

    if not frames:
        raise ValueError(f"trade files for {battery_id} contained no rows")
    return pd.concat(frames, ignore_index=True)


def _stats(pnls: np.ndarray, n_trials: int = 1) -> dict:
    pnls = np.asarray(pnls, dtype=float)
    pnls = pnls[np.isfinite(pnls)]
    if len(pnls) == 0:
        return {"n_trades": 0, "profit_factor": None, "sharpe": None, "dsr_ratio": None}
    dsr = gate.deflated_sharpe(pnls, n_trials=n_trials) if len(pnls) >= 8 else None
    return {
        "n_trades": len(pnls),
        "profit_factor": round(gate.profit_factor(pnls), 4),
        "sharpe": gate.sharpe(pnls),
        "dsr_ratio": (dsr or {}).get("ratio"),
    }


def year_pair_table(df: pd.DataFrame, combo_idx: int, n_trials: int = 1) -> pd.DataFrame:
    """One row per (year, pair) for a single combo."""
    sub = df[df["combo_idx"] == combo_idx]
    rows = []
    for (year, pair), g in sub.groupby(["year", "pair"]):
        rows.append({"year": year, "pair": pair, **_stats(g["pnl"].to_numpy(), n_trials)})
    return pd.DataFrame(rows).sort_values(["year", "pair"])


def year_pooled_table(df: pd.DataFrame, combo_idx: int, n_trials: int) -> pd.DataFrame:
    """One row per year, pooled across pairs, WITH the clustering caveat.

    n_trials is the full search size -- a per-year pooled number still paid
    for the whole grid search, so it is deflated accordingly.
    """
    sub = df[df["combo_idx"] == combo_idx]
    rows = []
    for year, g in sub.groupby("year"):
        st = _stats(g["pnl"].to_numpy(), n_trials)
        n = len(g)
        distinct_days = g["ts"].dt.normalize().nunique()
        st.update({
            "year": year,
            "distinct_session_days": distinct_days,
            "day_cluster_ratio": round(n / distinct_days, 2) if distinct_days else None,
        })
        rows.append(st)
    cols = ["year", "n_trades", "distinct_session_days", "day_cluster_ratio",
            "profit_factor", "sharpe", "dsr_ratio"]
    return pd.DataFrame(rows)[cols].sort_values("year")


def cross_pair_correlation_by_year(df: pd.DataFrame, combo_idx: int) -> dict:
    """How the pair-to-pair PnL relationship shifted year to year.

    Daily-summed PnL per pair, then the pairwise correlation matrix per
    year. A rising mean |correlation| means the pairs are increasingly the
    SAME bet (a dollar move), which is exactly when a pooled trade count
    overstates independent evidence.
    """
    sub = df[df["combo_idx"] == combo_idx].copy()
    sub["day"] = sub["ts"].dt.normalize()
    out = {}
    for year, g in sub.groupby("year"):
        daily = g.groupby(["day", "pair"])["pnl"].sum().unstack(fill_value=0.0)
        if daily.shape[1] < 2 or len(daily) < 5:
            continue
        corr = daily.corr()
        vals = corr.to_numpy()
        off = vals[~np.eye(len(vals), dtype=bool)]
        out[int(year)] = {
            "n_days": int(len(daily)),
            "n_pairs": int(daily.shape[1]),
            "mean_abs_corr": round(float(np.nanmean(np.abs(off))), 4),
            "max_corr": round(float(np.nanmax(off)), 4),
            "min_corr": round(float(np.nanmin(off)), 4),
        }
    return out


def build_report(battery_id: str, combo_idx: int | None = None, n_trials: int | None = None) -> dict:
    reg = json.loads((config.EXPERIMENTS / "registered" / f"{battery_id}.json").read_text(encoding="utf-8"))
    n_trials = n_trials or reg.get("n_trials", 1)

    # Rank combos by streaming over two columns, THEN read back only the
    # winner's rows. Ranking first and loading second is what keeps a 107M-row
    # log inside memory (see best_combo_streaming).
    if combo_idx is None:
        combo_idx = best_combo_streaming(battery_id)
    df = load_trades(battery_id, combo_idx=combo_idx)

    yp = year_pair_table(df, combo_idx, n_trials=1)
    yr = year_pooled_table(df, combo_idx, n_trials=n_trials)
    corr = cross_pair_correlation_by_year(df, combo_idx)

    # Per-year consistency: what fraction of years had PF > 1? A real edge
    # should not be one year carrying every other year.
    pf = yr["profit_factor"].dropna()
    consistency = {
        "n_years": int(len(pf)),
        "n_years_pf_above_1": int((pf > 1).sum()),
        "frac_years_positive": round(float((pf > 1).mean()), 3) if len(pf) else None,
    }

    report = {
        "battery_id": battery_id,
        "combo_idx": combo_idx,
        "n_trials_charged": n_trials,
        "year_pooled": yr.to_dict(orient="records"),
        "year_consistency": consistency,
        "cross_pair_correlation_by_year": corr,
        "year_pair": yp.to_dict(orient="records"),
    }
    out = config.SCORECARDS / f"{battery_id}_regime_table.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["_written_to"] = str(out)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("battery_id")
    ap.add_argument("--combo-idx", type=int, default=None)
    args = ap.parse_args()

    rep = build_report(args.battery_id, combo_idx=args.combo_idx)
    print(f"battery={rep['battery_id']} combo_idx={rep['combo_idx']} "
          f"n_trials_charged={rep['n_trials_charged']}\n")
    print("== per-year (pooled across pairs) ==")
    print(pd.DataFrame(rep["year_pooled"]).to_string(index=False))
    print("\n== year consistency ==")
    print(json.dumps(rep["year_consistency"], indent=2))
    print("\n== cross-pair daily-PnL correlation by year ==")
    print(json.dumps(rep["cross_pair_correlation_by_year"], indent=2))
    print(f"\nwritten to {rep['_written_to']}")
