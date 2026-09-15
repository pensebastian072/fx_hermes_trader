"""B02: MTF-filtered mean reversion, 7 USD majors, per-year folds 2015-2022.

HYPOTHESIS (pre-registered before any B02 result exists):
  A 1-minute mean-reversion fade is a bet that a move is noise. That bet
  should be BETTER when the higher timeframes disagree with each other (a
  range/chop regime, nothing structural driving the move) and WORSE when
  they strongly agree (a real trend — fading it gets run over). So gating
  entries on |multi-timeframe confluence| <= threshold should beat the
  unfiltered version.

DESIGN CHOICES AND WHY (all fixed in advance):
  * Universe = the 7 USD majors. Chosen on an EX-ANTE liquidity ground
    (deepest books / tightest spreads by BIS turnover), NOT because they
    backtested well in B01 — B01's per-pair table was a 14/14 coin flip
    with no majors/crosses split worth harvesting. Narrowing a universe
    after seeing results is the p-hacking move this note exists to rule out.
  * `confluence_max = 1.01` is included as an explicit CONTROL ARM: at that
    threshold the gate is always open, so the run contains its own
    unfiltered baseline. The question "does MTF add anything?" is then a
    within-battery comparison, not a cross-battery one against B01 (which
    used a different universe and period and so could not answer it).
  * Folds are YEARS, reported per year, never pooled into one number
    (box rule: a per-fold table beats a pooled number a sample skew could
    have manufactured). 2015-2022 spans CHF unpeg, 2016 Brexit/USD surge,
    2018 carry unwind, 2020 covid, 2021-22 divergence + BoJ suppression.
  * 2023+ stays LOCKED as holdout. Nothing here touches it.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b02_mtf_confluence --smoke
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b02_mtf_confluence --full
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from gpu_meanrev import config
from gpu_meanrev.backtest.batch_walkforward import run_fold
from gpu_meanrev.data.loader import build_panel
from gpu_meanrev.experiments import registry
from gpu_meanrev.features.multiframe import build_mtf_context, confluence_score
from gpu_meanrev.features.rolling import rolling_bollinger_percent_b, rolling_zscore
from gpu_meanrev.gpu import get_device, set_seed
from gpu_meanrev.signals.mean_reversion import Combo, combo_grid

BATTERY_ID = "B02_mtf_confluence"
DATASET_FAMILY = "fx_1min_meanrev"

HYPOTHESIS = (
    "1-minute FX mean-reversion fades are more profitable when higher "
    "timeframes (15m/1h/4h/1d) DISAGREE on trend direction (range regime) "
    "than when they strongly agree (trend regime). Gating entries on "
    "|confluence| <= threshold should beat the unfiltered control arm "
    "(confluence_max=1.01) on the 7 USD majors, reported per year 2015-2022. "
    "Pure price-action only -- no macro/news features."
)

# 7 USD majors -- top BIS turnover, an ex-ante liquidity choice.
MAJORS_7 = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]

MTF_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

# 4 x 3 x 2 x 3 = 72 combos, fixed before any result exists.
GRID_B02 = {
    "lookback_min": [20, 60, 120, 240],
    "entry_z": [1.5, 2.0, 2.5],
    "exit_rule": ["inner_band", "max_hold"],
    "session": ["24h"],
    "signal_family": ["zscore_only"],
    "confluence_max": [0.34, 0.67, 1.01],  # 1.01 = control arm, gate always open
}

YEARS = list(range(2015, 2023))  # 2015-2022 inclusive; 2023+ is locked holdout


def _build_confluence(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-pair multi-timeframe confluence score, aligned to the 1m index.

    Leak guard lives in features/multiframe.py (bars stamped at close, then
    shifted one bar) and is pinned by tests/test_gpu_meanrev_multiframe.py.
    """
    out = {}
    for pair in panel.columns:
        closes = panel[pair].dropna()
        mtf = build_mtf_context(closes, timeframes=MTF_TIMEFRAMES)
        out[pair] = confluence_score(mtf).reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


def _confluence_gate(conf: torch.Tensor, combos: list[Combo], device) -> torch.Tensor:
    """(n_pairs, n_bars) confluence -> (n_pairs*n_combos, n_bars) bool gate.

    NaN confluence (not enough higher-timeframe history yet) compares False,
    so no entry is taken before the MTF context actually exists — the
    conservative direction.
    """
    n_pairs, n_bars = conf.shape
    n_combos = len(combos)
    thresholds = torch.tensor([c.confluence_max for c in combos],
                              device=device, dtype=torch.float32).view(1, n_combos, 1)
    conf_exp = conf.to(device).unsqueeze(1)               # (n_pairs, 1, n_bars)
    gate = conf_exp.abs() <= thresholds                     # broadcast -> (n_pairs, n_combos, n_bars)
    return gate.reshape(n_pairs * n_combos, n_bars)


def run_year(year: int, pairs: list[str], combos: list[Combo], block_days: int,
             combo_chunk_size: int, device) -> list[dict]:
    """All trades for one calendar year."""
    # Bound BOTH ends before the union/ffill — loading year..2023 to use one
    # year is what drove the box into swap on 2026-08-05.
    panel = build_panel(pairs, start_date=f"{year}-01-01", end_date=f"{year + 1}-01-01")
    if panel.empty:
        return []

    conf_df = _build_confluence(panel)
    windows = sorted(set(c.lookback_min for c in combos))
    cost = torch.tensor([config.cost_per_side(p) for p in pairs], dtype=torch.float32)

    year_trades: list[dict] = []
    cur = panel.index[0]
    last = panel.index[-1]
    while cur <= last:
        nxt = cur + pd.Timedelta(days=block_days)
        blk = panel.loc[cur:min(nxt, last + pd.Timedelta(minutes=1))]
        cur = nxt
        if len(blk) < max(windows) + 1:
            continue

        prices_cpu = torch.tensor(blk.to_numpy(dtype="float32").T)
        conf_blk = torch.tensor(conf_df.loc[blk.index].to_numpy(dtype="float32").T)
        z_by = {w: rolling_zscore(prices_cpu, w).float() for w in windows}
        bb_by = {w: rolling_bollinger_percent_b(prices_cpu, w).float() for w in windows}

        for i in range(0, len(combos), combo_chunk_size):
            chunk = combos[i:i + combo_chunk_size]
            gate = _confluence_gate(conf_blk, chunk, device)
            _pnl, trades = run_fold(
                prices=prices_cpu, z_by_window=z_by, bb_by_window=bb_by,
                fold_index=blk.index, combos=chunk, pairs=pairs,
                cost_per_side=cost, device=device, collect_trades=True,
                extra_entry_gate=gate,
            )
            for t in trades:
                t["combo_idx"] += i
                t["year"] = year
            year_trades.extend(trades)
    return year_trades


def run_battery(battery_id: str = BATTERY_ID, pairs: list[str] | None = None,
                grid: dict | None = None, years: list[int] | None = None,
                block_days: int = 30, combo_chunk_size: int = 24,
                register: bool = True) -> dict:
    pairs = pairs or MAJORS_7
    grid = grid or GRID_B02
    years = years or YEARS
    device = get_device()
    set_seed(config.SEED)
    t0 = time.time()

    combos = combo_grid(grid)
    if register:
        try:
            registry.register(
                battery_id=battery_id, hypothesis=HYPOTHESIS,
                dataset_family=DATASET_FAMILY, pairs=pairs,
                param_grid=grid, n_trials=len(combos),
            )
        except FileExistsError:
            pass

    ckpt_dir = config.DATA_REPORTS / f"{battery_id}_checkpoint" / "trades"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        out_path = ckpt_dir / f"year_{year}.parquet"
        if out_path.exists():
            continue  # resumable: never recompute a finished year
        trades = run_year(year, pairs, combos, block_days, combo_chunk_size, device)
        pd.DataFrame(trades).to_parquet(out_path)

    parts = sorted(ckpt_dir.glob("year_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()

    return {
        "battery_id": battery_id, "n_pairs": len(pairs), "n_combos": len(combos),
        "years": years, "n_trades_total": len(df),
        "elapsed_sec": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        out = run_battery(
            battery_id="B02_mtf_confluence_SMOKE",
            pairs=["EURUSD", "USDJPY"],
            grid={**GRID_B02, "lookback_min": [60], "entry_z": [2.0], "exit_rule": ["inner_band"]},
            years=[2022], block_days=30, combo_chunk_size=8, register=False,
        )
        print(json.dumps(out, indent=2, default=str))
    elif args.full:
        out = run_battery()
        print(json.dumps(out, indent=2, default=str))
    else:
        ap.print_help()
