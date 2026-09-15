"""Pre-registered parameter grid -> Combo objects, plus session filtering.

The entry/exit STATE MACHINE (path-dependent: an open trade's exit depends
on prior bars) lives in backtest/batch_walkforward.py — this module only
builds the static per-combo parameters and the session-time mask, both of
which are the same for every fold.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from gpu_meanrev import config


@dataclass(frozen=True)
class Combo:
    lookback_min: int
    entry_z: float
    exit_rule: str        # "inner_band" | "max_hold"
    session: str           # "24h" | "london_ny_overlap"
    signal_family: str    # "zscore_only" | "zscore_bollinger_confirm"
    # B02 only. Max |multi-timeframe confluence| at which a fade is still
    # allowed: low = only fade when higher timeframes DISAGREE (range
    # regime), 1.01 = no filter at all (the control arm). Defaulted so B01's
    # grid and its already-written registration are unaffected.
    confluence_max: float = 1.01
    # B03 only. Explicit holding period in BARS, decoupling the hold from the
    # lookback. B01/B02 tied them together as 4x by assertion, and the
    # 2026-08-06 post-mortem found no reversion half-life justifying that
    # ratio. None keeps the old behaviour exactly, so B01/B02 are unaffected.
    hold_bars: int | None = None

    @property
    def max_hold_bars(self) -> int:
        return self.hold_bars if self.hold_bars is not None else 4 * self.lookback_min

    def as_dict(self) -> dict:
        return {
            "lookback_min": self.lookback_min, "entry_z": self.entry_z,
            "exit_rule": self.exit_rule, "session": self.session,
            "signal_family": self.signal_family,
            "confluence_max": self.confluence_max,
        }


def combo_grid(grid: dict | None = None) -> list[Combo]:
    """Every combination in the pre-registered grid (config.GRID by default).

    `confluence_max` is optional in the grid dict — absent means B01-style
    (no MTF filter), so an existing grid keeps producing an identical combo
    list.
    """
    grid = grid or config.GRID
    conf_values = grid.get("confluence_max", [1.01])
    combos = []
    for lb, ez, ex, ses, fam, conf in itertools.product(
        grid["lookback_min"], grid["entry_z"], grid["exit_rule"],
        grid["session"], grid["signal_family"], conf_values,
    ):
        combos.append(Combo(lb, ez, ex, ses, fam, conf))
    return combos


def session_mask(index, session: str) -> torch.Tensor:
    """(n_bars,) bool tensor — True where new entries are allowed at that bar.

    "24h" allows entries at any bar; "london_ny_overlap" restricts entries
    to 13:00-16:00 UTC (does not force an exit outside that window — an
    open position still runs its normal exit rule regardless of session).
    """
    if session == "24h":
        return torch.ones(len(index), dtype=torch.bool)
    hours = index.hour
    mask = (hours >= 13) & (hours < 16)
    return torch.tensor(mask.to_numpy() if hasattr(mask, "to_numpy") else mask, dtype=torch.bool)
