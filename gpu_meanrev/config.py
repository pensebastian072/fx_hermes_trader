"""Shared config: paths, pair universe, grid, holdout, gate location.

GPU/CUDA 1-minute FX mean-reversion study (2026-08). Research/advisory only,
paper-first repo (see fx_hermes_trader/CLAUDE.md) — nothing here places an
order or touches execution/risk. Bypasses hermes/ entirely: this is a pure
backtest path, not the live webhook flow. Every result stays SHADOW until the
canonical overfit gate clears (see gate.py — imported from macro_gpu_lab,
never re-ported).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATA = REPO / "data"
RAW = DATA / "raw" / "histdata_fx_1m"
FEATURES_CACHE = DATA / "features" / "histdata_fx_1m"

# Research journal (committed): experiments, scorecards, logs. Separate from
# reports/experiments/ (gitignored, existing paper-backtest report output —
# a different provenance mechanism, do not conflate).
JOURNAL = REPO / "journal"
EXPERIMENTS = JOURNAL / "experiments"
SCORECARDS = JOURNAL / "scorecards"
DATA_REPORTS = JOURNAL / "data"

# HF dataset: 159 FX/commodity/index pairs, 1-min OHLCV parquet, UTC, free/no-auth.
HF_REPO_ID = "elthariel/histdata_fx_1m"
HF_REPO_TYPE = "dataset"

# Candidate research universe: majors + common crosses. Resolved against the
# real HF catalog by data/hf_catalog.py -> configs/research/fx_1m_pairs.yaml
# (some thin crosses may not exist in the 159-symbol set; land honestly).
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
CROSSES_CANDIDATE = [
    "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
    "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
    "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
    "NZDJPY", "NZDCHF", "NZDCAD",
    "CADJPY", "CADCHF", "CHFJPY",
]
PAIRS_CANDIDATE = MAJORS + CROSSES_CANDIDATE  # 28 candidates
RESOLVED_PAIRS_FILE = REPO / "configs" / "research" / "fx_1m_pairs.yaml"

# Probe pairs (Stage 1) — 2 majors + 1 cross, before committing to the full pull.
PROBE_PAIRS = ["EURUSD", "USDJPY", "EURGBP"]

# Never forward-fill across a gap longer than this many minutes when building
# the common cross-pair UTC index (data/loader.py). Set from Stage 1's real
# probe (journal/data/probe_report.json, 2026-08-04): gaps.parquet's "length"
# column is in SECONDS and excludes weekend closures already (max observed
# gap 9967s/~166min, not a ~48h weekend span). Distribution: median 91s,
# 90th pct 204s, 99th pct 569s, 99.9th pct 3594s(~60min). 10 minutes bridges
# ~99% of real within-session gaps; anything longer is flagged NaN rather
# than carried forward, so a genuine dead spot can't manufacture a fake
# reversion signal or fake cross-pair correlation.
MAX_FORWARD_FILL_MINUTES: int = 10

# Pre-registered parameter grid (fixed BEFORE running, quant-research-gate
# rule — no post-hoc expansion). 5 x 4 x 2 x 2 x 2 = 160 combos.
GRID = {
    "lookback_min": [20, 60, 120, 240, 480],
    "entry_z": [1.5, 2.0, 2.5, 3.0],
    "exit_rule": ["inner_band", "max_hold"],       # inner |z|<0.5, or 4x lookback bars
    "session": ["24h", "london_ny_overlap"],        # overlap = 13:00-16:00 UTC
    "signal_family": ["zscore_only", "zscore_bollinger_confirm"],
}

# Round-trip cost model: FRACTION of price per side (spread/2 + slippage).
# Hard-won lesson (fx_hermes v3, see alpaca_gpu_lab/src/config.py): costs are
# fractions, never absolute units. Majors trade tighter than crosses.
COST_PER_SIDE = {"major": 0.00005, "cross": 0.00015}  # 0.5bp / 1.5bp


def cost_per_side(pair: str) -> float:
    tier = "major" if pair in MAJORS else "cross"
    return COST_PER_SIDE[tier]


# ---------------------------------------------------------------------------
# Holdout lock: unlike alpaca_gpu_lab's fixed 2026-01-01, this archive's real
# end date had to be measured (histdata.com mirrors lag real-time). Measured
# 2026-08-04 via data/loader.py::common_date_range() across all 28 resolved
# pairs: common coverage is 2008-03-30 -> 2025-03-21 (~17.0 years; the 2008
# start is set by the latest-starting pair, GBPNZD/AUDCHF/etc.). 2023-01-01
# holds out the final ~2.2 years (~13% of the common window), in the 10-15%
# target range. Same lock mechanism as alpaca_gpu_lab: the default loader
# clips >= HOLDOUT_START; reading it needs both unlock_holdout=True in code
# AND FX_MEANREV_HOLDOUT_UNLOCK=yes, flipped once, at the very end.
# ---------------------------------------------------------------------------
HOLDOUT_START: str = "2023-01-01"
HOLDOUT_UNLOCK = os.environ.get("FX_MEANREV_HOLDOUT_UNLOCK", "no").lower() in ("yes", "true", "1")

# Canonical overfit gate lives in macro_gpu_lab (see gate.py). Thresholds
# (PBO_MAX, DEFLATED_SHARPE_MIN) live there too — never duplicated here.
MACRO_GPU_LAB_DIR = Path(os.environ.get("MACRO_GPU_LAB_DIR", r"C:\Users\<your-user>\macro_gpu_lab"))

SEED = 42
