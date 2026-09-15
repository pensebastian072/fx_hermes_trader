"""B02's confluence entry-gate, and that it leaves B01's behaviour untouched.

The whole B02 comparison rests on the control arm (confluence_max=1.01)
being *exactly* the unfiltered strategy — if the gate quietly altered the
control too, "filtered beats unfiltered" would be unfalsifiable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from gpu_meanrev.backtest.batch_walkforward import run_fold
from gpu_meanrev.batteries.b02_mtf_confluence import _confluence_gate
from gpu_meanrev.features.rolling import rolling_bollinger_percent_b, rolling_zscore
from gpu_meanrev.signals.mean_reversion import Combo, combo_grid

CPU = torch.device("cpu")


def _scenario(n=200, window=20):
    base = 1.10
    prices = np.full(n, base)
    spike = window + 10
    prices[spike:spike + 3] = base * 1.02
    prices[spike + 3:] = base
    idx = pd.date_range("2020-01-06", periods=n, freq="min", tz="UTC")
    t = torch.tensor(prices, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(t.double(), window).float()
    bb = rolling_bollinger_percent_b(t.double(), window).float()
    return t, z, bb, idx, window


def test_control_arm_matches_no_gate_exactly():
    t, z, bb, idx, window = _scenario()
    combo = Combo(window, 1.5, "inner_band", "24h", "zscore_only", confluence_max=1.01)

    # conf = 0 everywhere -> |0| <= 1.01 -> gate fully open
    conf = torch.zeros(1, t.shape[1])
    gate = _confluence_gate(conf, [combo], CPU)

    pnl_gated, tr_gated = run_fold(
        prices=t, z_by_window={window: z}, bb_by_window={window: bb}, fold_index=idx,
        combos=[combo], pairs=["TEST"], cost_per_side=torch.tensor([0.0]),
        device=CPU, extra_entry_gate=gate,
    )
    pnl_plain, tr_plain = run_fold(
        prices=t, z_by_window={window: z}, bb_by_window={window: bb}, fold_index=idx,
        combos=[combo], pairs=["TEST"], cost_per_side=torch.tensor([0.0]), device=CPU,
    )
    assert pnl_gated.item() == pytest.approx(pnl_plain.item(), abs=1e-9)
    assert len(tr_gated) == len(tr_plain)


def test_closed_gate_blocks_all_entries():
    t, z, bb, idx, window = _scenario()
    combo = Combo(window, 1.5, "inner_band", "24h", "zscore_only", confluence_max=0.34)

    # conf = 1.0 everywhere (all timeframes agree) -> |1.0| > 0.34 -> gate shut
    conf = torch.ones(1, t.shape[1])
    gate = _confluence_gate(conf, [combo], CPU)

    pnl, trades = run_fold(
        prices=t, z_by_window={window: z}, bb_by_window={window: bb}, fold_index=idx,
        combos=[combo], pairs=["TEST"], cost_per_side=torch.tensor([0.0]),
        device=CPU, extra_entry_gate=gate,
    )
    assert pnl.item() == 0.0
    assert trades == []


def test_nan_confluence_blocks_entry():
    """No higher-timeframe history yet must mean no trade, not a free trade."""
    t, z, bb, idx, window = _scenario()
    combo = Combo(window, 1.5, "inner_band", "24h", "zscore_only", confluence_max=1.01)
    conf = torch.full((1, t.shape[1]), float("nan"))
    gate = _confluence_gate(conf, [combo], CPU)
    assert not gate.any(), "NaN confluence must compare False, never open the gate"


def test_gate_shape_mismatch_raises():
    t, z, bb, idx, window = _scenario()
    combo = Combo(window, 1.5, "inner_band", "24h", "zscore_only")
    bad = torch.ones(3, 7, dtype=torch.bool)
    with pytest.raises(ValueError, match="extra_entry_gate must be"):
        run_fold(prices=t, z_by_window={window: z}, bb_by_window={window: bb},
                 fold_index=idx, combos=[combo], pairs=["TEST"],
                 cost_per_side=torch.tensor([0.0]), device=CPU, extra_entry_gate=bad)


def test_b01_grid_unchanged_by_new_field():
    """A grid with no confluence_max must yield the same 160 B01 combos."""
    from gpu_meanrev import config
    combos = combo_grid(config.GRID)
    assert len(combos) == 160
    assert all(c.confluence_max == 1.01 for c in combos)


def test_b02_grid_size_and_control_arm_present():
    from gpu_meanrev.batteries.b02_mtf_confluence import GRID_B02
    combos = combo_grid(GRID_B02)
    assert len(combos) == 72
    # 3 confluence levels x 24 = 72, so the unfiltered control is a third of
    # the grid — enough arms to compare filtered vs unfiltered within-battery.
    assert sum(1 for c in combos if c.confluence_max == 1.01) == 24
