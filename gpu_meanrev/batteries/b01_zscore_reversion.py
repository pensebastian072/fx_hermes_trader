"""B01: 1-minute FX z-score / Bollinger %B mean reversion, pre-registered grid.

Pure price-action (no macro/news/rate features this round). Fixed, pre-
registered parameters -- nothing is fit on any subset of the data, so there
is no train/test leakage risk in the usual walk-forward sense. Time is still
chunked into blocks purely for GPU memory reasons (see _time_blocks): the
batch dimension is n_pairs x n_combos-in-chunk, and a full year of 1-minute
bars x a wide combo chunk does not fit in the RTX 3050's 6GB otherwise.
Position state resets to flat at each block boundary -- a documented
simplification (see module-level NOTE) rather than a hidden leak.

Checkpointed/resumable (probe-before-long-run discipline): each completed
block's trades land in journal/data/<battery_id>_checkpoint/trades/, and
completed_blocks.json tracks progress -- a killed run resumes by skipping
already-done blocks, never re-computing them.

CLI:
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b01_zscore_reversion --smoke
  .venv\\Scripts\\python.exe -m gpu_meanrev.batteries.b01_zscore_reversion --full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from gpu_meanrev import config
from gpu_meanrev.backtest.batch_walkforward import run_fold
from gpu_meanrev.data.loader import build_panel
from gpu_meanrev.experiments import registry
from gpu_meanrev.features.rolling import rolling_bollinger_percent_b, rolling_zscore
from gpu_meanrev.gpu import get_device, set_seed
from gpu_meanrev.reporting.scorecard import build_and_write_scorecard
from gpu_meanrev.signals.mean_reversion import Combo, combo_grid

HYPOTHESIS = (
    "1-minute FX price shows short-horizon mean reversion detectable via a "
    "rolling z-score (and optionally Bollinger %B confirmation), net of "
    "realistic per-side costs, across FX majors + common crosses. Pure "
    "price-action only -- no macro/news/rate features this round."
)

DATASET_FAMILY = "fx_1min_meanrev"

# NOTE (documented simplification): position state resets to flat at each
# time-block boundary. Lookback windows are <=480 minutes (8h); blocks are
# months long -- boundary truncation affects a handful of bars out of
# hundreds of thousands per block. Not zero-effect, but not hidden either.


def _load_pairs() -> list[str]:
    return yaml.safe_load(config.RESOLVED_PAIRS_FILE.read_text(encoding="utf-8"))["pairs"]


def _time_blocks(index: pd.DatetimeIndex, block_days: int):
    if len(index) == 0:
        return
    start = index[0]
    end = index[-1]
    cur = start
    while cur <= end:
        nxt = cur + pd.Timedelta(days=block_days)
        yield cur, min(nxt, end + pd.Timedelta(minutes=1))
        cur = nxt


def _combo_chunks(combos: list[Combo], chunk_size: int):
    for i in range(0, len(combos), chunk_size):
        yield combos[i:i + chunk_size]


class Checkpoint:
    """Per-battery resumable progress: completed block indices + per-block
    trade parquet files. Never re-computes a block already on disk."""

    def __init__(self, battery_id: str):
        self.dir = config.DATA_REPORTS / f"{battery_id}_checkpoint"
        self.trades_dir = self.dir / "trades"
        self.state_path = self.dir / "state.json"
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {
            "completed_blocks": [], "n_trades_total": 0,
        }

    def is_done(self, block_idx: int) -> bool:
        return block_idx in self.state["completed_blocks"]

    def save_block(self, block_idx: int, trades: list[dict]) -> None:
        if trades:
            pd.DataFrame(trades).to_parquet(self.trades_dir / f"block_{block_idx:05d}.parquet")
        self.state["completed_blocks"].append(block_idx)
        self.state["n_trades_total"] += len(trades)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def load_all_trades(self) -> pd.DataFrame:
        parts = sorted(self.trades_dir.glob("block_*.parquet"))
        if not parts:
            return pd.DataFrame(columns=["pair", "combo_idx", "pnl", "ts", "block"])
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def run_battery(
    battery_id: str,
    pairs: list[str],
    grid: dict,
    block_days: int,
    combo_chunk_size: int,
    register: bool = True,
    max_blocks: int | None = None,
    use_checkpoint: bool = True,
    start_date: str | None = None,
) -> dict:
    device = get_device()
    set_seed(config.SEED)
    t_start = time.time()

    combos = combo_grid(grid)
    n_trials_this_run = len(combos)

    if register:
        try:
            registry.register(
                battery_id=battery_id, hypothesis=HYPOTHESIS,
                dataset_family=DATASET_FAMILY, pairs=pairs,
                param_grid=grid, n_trials=n_trials_this_run,
            )
        except FileExistsError:
            pass  # already registered — append-only, fine to re-run

    panel = build_panel(pairs, start_date=start_date)  # holdout-clipped by default
    cost_per_side = torch.tensor([config.cost_per_side(p) for p in pairs], dtype=torch.float32)
    windows = sorted(set(c.lookback_min for c in combos))

    ckpt = Checkpoint(battery_id) if use_checkpoint else None
    blocks = list(_time_blocks(panel.index, block_days))
    if max_blocks is not None:
        blocks = blocks[:max_blocks]

    n_skipped = 0
    for b_i, (blk_start, blk_end) in enumerate(blocks):
        if ckpt is not None and ckpt.is_done(b_i):
            n_skipped += 1
            continue
        blk = panel.loc[blk_start:blk_end]
        if len(blk) < max(windows) + 1:
            if ckpt is not None:
                ckpt.save_block(b_i, [])
            continue
        prices_cpu = torch.tensor(blk.to_numpy(dtype="float32").T)  # (n_pairs, n_bars)

        z_by_window = {w: rolling_zscore(prices_cpu, w).float() for w in windows}
        bb_by_window = {w: rolling_bollinger_percent_b(prices_cpu, w).float() for w in windows}

        block_trades = []
        for chunk in _combo_chunks(combos, combo_chunk_size):
            pnl_mat, trades = run_fold(
                prices=prices_cpu, z_by_window=z_by_window, bb_by_window=bb_by_window,
                fold_index=blk.index, combos=chunk, pairs=pairs,
                cost_per_side=cost_per_side, device=device, collect_trades=True,
            )
            base_idx = combos.index(chunk[0])
            for t in trades:
                t["combo_idx"] += base_idx
            block_trades.extend(trades)
            del pnl_mat  # aggregate PnL is re-derivable from trades; don't hold both

        for t in block_trades:
            t["block"] = b_i
        if ckpt is not None:
            ckpt.save_block(b_i, block_trades)

    elapsed = time.time() - t_start
    trades_df = ckpt.load_all_trades() if ckpt is not None else pd.DataFrame()

    winner_idx = None
    if not trades_df.empty:
        stats = trades_df.groupby("combo_idx")["pnl"].agg(["mean", "std", "count"])
        stats = stats[stats["count"] >= 8]
        if not stats.empty:
            stats["score"] = stats["mean"] / stats["std"].replace(0, float("nan"))
            winner_idx = int(stats["score"].idxmax())

    result = {
        "battery_id": battery_id, "n_pairs": len(pairs), "n_combos": len(combos),
        "n_blocks": len(blocks), "n_blocks_skipped_cached": n_skipped,
        "n_trades_total": len(trades_df), "elapsed_sec": round(elapsed, 2),
        "winner_combo_idx": winner_idx,
        "winner_combo": combos[winner_idx].as_dict() if winner_idx is not None else None,
    }

    if winner_idx is not None:
        winner_trades = trades_df[trades_df["combo_idx"] == winner_idx].sort_values("ts")
        pnl_by_pair = {p: winner_trades[winner_trades["pair"] == p]["pnl"].to_numpy() for p in pairs}
        scorecard = build_and_write_scorecard(
            battery_id=battery_id, combo_dict=combos[winner_idx].as_dict(),
            pnl_by_pair=pnl_by_pair, entry_timestamps=winner_trades["ts"],
            n_trials=n_trials_this_run,
        )
        result["scorecard"] = scorecard

    return result


def run_years(
    battery_id: str,
    pairs: list[str],
    grid: dict,
    years: list[int],
    block_days: int = 30,
    combo_chunk_size: int = 32,
) -> dict:
    """Year-at-a-time runner — the memory-safe replacement for the whole-panel loop.

    WHY THIS EXISTS (2026-08-05): `run_battery` builds ONE panel spanning the
    entire 2000-2023 archive (~1.8GB for the result, several times that in
    intermediates) and then slices 90-day blocks out of it. On this 16GB box
    that drove free memory to ~0.9GB and wedged the run: 25 minutes at block
    31/91 with the GPU at 7%, because it was swapping, not computing. Every
    resume paid that cost again.

    Here each year builds its own bounded panel (start_date AND end_date),
    so peak memory is ~1/15th and a resume only redoes the current year.
    Checkpoint unit is the YEAR (trades/year_<Y>.parquet), which is also
    what the per-year regime table wants anyway.

    Pre-existing block_*.parquet files from the old runner are still valid
    trades — reporting/regime_table.py reads both, using block files only
    for years the year-files don't cover, so nothing is double-counted.
    """
    device = get_device()
    set_seed(config.SEED)
    t0 = time.time()

    combos = combo_grid(grid)
    cost_per_side = torch.tensor([config.cost_per_side(p) for p in pairs], dtype=torch.float32)
    windows = sorted(set(c.lookback_min for c in combos))

    ckpt_dir = config.DATA_REPORTS / f"{battery_id}_checkpoint" / "trades"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    done, skipped = [], []
    for year in years:
        out_path = ckpt_dir / f"year_{year}.parquet"
        if out_path.exists():
            skipped.append(year)
            continue
        panel = build_panel(pairs, start_date=f"{year}-01-01", end_date=f"{year + 1}-01-01")
        if panel.empty:
            pd.DataFrame(columns=["bar", "ts", "pair", "combo_idx", "pnl"]).to_parquet(out_path)
            continue

        year_trades: list[dict] = []
        for blk_start, blk_end in _time_blocks(panel.index, block_days):
            blk = panel.loc[blk_start:blk_end]
            if len(blk) < max(windows) + 1:
                continue
            prices_cpu = torch.tensor(blk.to_numpy(dtype="float32").T)
            z_by = {w: rolling_zscore(prices_cpu, w).float() for w in windows}
            bb_by = {w: rolling_bollinger_percent_b(prices_cpu, w).float() for w in windows}
            for i in range(0, len(combos), combo_chunk_size):
                chunk = combos[i:i + combo_chunk_size]
                _pnl, trades = run_fold(
                    prices=prices_cpu, z_by_window=z_by, bb_by_window=bb_by,
                    fold_index=blk.index, combos=chunk, pairs=pairs,
                    cost_per_side=cost_per_side, device=device, collect_trades=True,
                )
                for t in trades:
                    t["combo_idx"] += i
                year_trades.extend(trades)
            del prices_cpu, z_by, bb_by
        pd.DataFrame(year_trades).to_parquet(out_path)
        done.append(year)
        del panel

    return {
        "battery_id": battery_id, "years_computed": done, "years_cached": skipped,
        "n_pairs": len(pairs), "n_combos": len(combos),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def _register_full_spec_only() -> None:
    """Pre-register the full battery's design WITHOUT running it — so both
    the pilot and the full run are committed to before either has results
    (quant-research-gate: no post-hoc grid changes after seeing a number)."""
    combos = combo_grid(config.GRID)
    try:
        registry.register(
            battery_id="B01_zscore_reversion", hypothesis=HYPOTHESIS,
            dataset_family=DATASET_FAMILY, pairs=_load_pairs(),
            param_grid=config.GRID, n_trials=len(combos),
        )
    except FileExistsError:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--years", action="store_true",
                    help="memory-safe year-at-a-time run (preferred over --full)")
    ap.add_argument("--from-year", type=int, default=2008)
    ap.add_argument("--to-year", type=int, default=2022)
    args = ap.parse_args()

    if args.smoke:
        smoke_grid = {
            "lookback_min": [20, 60],
            "entry_z": [2.0, 3.0],
            "exit_rule": ["inner_band"],
            "session": ["24h"],
            "signal_family": ["zscore_only"],
        }
        out = run_battery(
            battery_id="B01_zscore_reversion_SMOKE",
            pairs=config.PROBE_PAIRS,
            grid=smoke_grid,
            block_days=90,
            combo_chunk_size=64,
            register=False,
            max_blocks=1,
            use_checkpoint=False,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "scorecard"}, indent=2, default=str))
    elif args.pilot:
        # Same grid/design as the full run, narrowed to the most recent ~4
        # in-sample months (2022-09-01 -> holdout 2023-01-01) so it finishes
        # in ~1-2hr instead of ~83hr. Registered as a DISTINCT battery id —
        # not a substitute for the full run, a separate honest pre-registered
        # pilot. The full spec is pre-registered here too (unrun) so neither
        # design is chosen after seeing the other's results.
        _register_full_spec_only()
        out = run_battery(
            battery_id="B01_zscore_reversion_PILOT",
            pairs=_load_pairs(),
            grid=config.GRID,
            block_days=30,
            combo_chunk_size=32,
            register=True,
            start_date="2022-09-01",
        )
        print(json.dumps({k: v for k, v in out.items() if k != "scorecard"}, indent=2, default=str))
    elif args.years:
        _register_full_spec_only()
        out = run_years(
            battery_id="B01_zscore_reversion",
            pairs=_load_pairs(),
            grid=config.GRID,
            years=list(range(args.from_year, args.to_year + 1)),
        )
        print(json.dumps(out, indent=2, default=str))
    elif args.full:
        out = run_battery(
            battery_id="B01_zscore_reversion",
            pairs=_load_pairs(),
            grid=config.GRID,
            block_days=90,
            combo_chunk_size=32,
            register=True,
        )
        print(json.dumps({k: v for k, v in out.items() if k != "scorecard"}, indent=2, default=str))
    else:
        ap.print_help()
