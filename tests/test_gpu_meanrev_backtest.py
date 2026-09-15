"""Correctness checks for the GPU-batched mean-reversion state machine.

batch_walkforward.run_fold is the one genuinely new, path-dependent piece of
this study — worth pinning down with a synthetic, hand-verifiable scenario
before trusting it on real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from gpu_meanrev.backtest.batch_walkforward import run_fold
from gpu_meanrev.features.rolling import rolling_bollinger_percent_b, rolling_zscore
from gpu_meanrev.signals.mean_reversion import Combo


def _spike_and_revert_prices(n=200, base=1.10, window=20):
    """Flat baseline, then a sharp spike, then reversion back to baseline."""
    prices = np.full(n, base, dtype=np.float64)
    spike_start = window + 10
    prices[spike_start:spike_start + 3] = base * 1.02  # +2% spike
    # revert back to baseline over the next few bars
    prices[spike_start + 3:] = base
    return prices


def _fold_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-06", periods=n, freq="min", tz="UTC")  # a Monday


def test_zscore_reversion_produces_a_profitable_short():
    """A price that spikes then reverts should trigger a short that profits."""
    window = 20
    n = 200
    prices_np = _spike_and_revert_prices(n=n, window=window)
    idx = _fold_index(n)

    prices = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(0)  # (1, n)
    z = rolling_zscore(prices.double(), window).float()
    bb = rolling_bollinger_percent_b(prices.double(), window).float()

    combo = Combo(lookback_min=window, entry_z=1.5, exit_rule="inner_band",
                  session="24h", signal_family="zscore_only")

    cost = torch.tensor([0.0])  # isolate the reversion PnL from cost model in this check
    pnl, trades = run_fold(
        prices=prices,
        z_by_window={window: z},
        bb_by_window={window: bb},
        fold_index=idx,
        combos=[combo],
        pairs=["TEST"],
        cost_per_side=cost,
        device=torch.device("cpu"),
    )
    assert pnl.shape == (1, 1)
    assert pnl.item() > 0, "a genuine spike-then-revert should profit a short entry"
    assert len(trades) >= 1
    assert all(t["pair"] == "TEST" and t["combo_idx"] == 0 for t in trades)
    assert sum(t["pnl"] for t in trades) == pytest.approx(pnl.item(), abs=1e-5), \
        "per-trade records must sum to the aggregate fold PnL"


def test_flat_series_never_trades():
    """A dead-flat series has std=0 everywhere -> z=0 always -> never triggers."""
    window = 20
    n = 100
    prices_np = np.full(n, 1.10, dtype=np.float64)
    idx = _fold_index(n)

    prices = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(prices.double(), window).float()
    bb = rolling_bollinger_percent_b(prices.double(), window).float()

    combo = Combo(lookback_min=window, entry_z=1.5, exit_rule="inner_band",
                  session="24h", signal_family="zscore_only")
    pnl, _trades = run_fold(
        prices=prices, z_by_window={window: z}, bb_by_window={window: bb},
        fold_index=idx, combos=[combo], pairs=["TEST"],
        cost_per_side=torch.tensor([0.0001]), device=torch.device("cpu"),
    )
    assert pnl.item() == 0.0


def test_cost_model_reduces_pnl():
    """Same scenario with nonzero costs must produce strictly less PnL."""
    window = 20
    n = 200
    prices_np = _spike_and_revert_prices(n=n, window=window)
    idx = _fold_index(n)
    prices = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(prices.double(), window).float()
    bb = rolling_bollinger_percent_b(prices.double(), window).float()
    combo = Combo(lookback_min=window, entry_z=1.5, exit_rule="inner_band",
                  session="24h", signal_family="zscore_only")

    pnl_free, _t1 = run_fold(prices=prices, z_by_window={window: z}, bb_by_window={window: bb},
                         fold_index=idx, combos=[combo], pairs=["TEST"],
                         cost_per_side=torch.tensor([0.0]), device=torch.device("cpu"))
    pnl_costly, _t2 = run_fold(prices=prices, z_by_window={window: z}, bb_by_window={window: bb},
                           fold_index=idx, combos=[combo], pairs=["TEST"],
                           cost_per_side=torch.tensor([0.0005]), device=torch.device("cpu"))
    assert pnl_costly.item() < pnl_free.item()


def test_max_hold_exit_rule_forces_exit():
    """max_hold combo must exit by max_hold_bars even if z hasn't reverted."""
    window = 20
    n = 300
    # Flat, then a sustained RAMP (not a one-step jump to a new flat level --
    # that would make entry_price == forced-exit price and prove nothing).
    # A short entered early in the ramp, held to max_hold, forced-closed
    # while price is still climbing -> unambiguous loss.
    prices_np = np.full(n, 1.10, dtype=np.float64)
    ramp_start = window + 5
    ramp_len = n - ramp_start
    prices_np[ramp_start:] = 1.10 + np.linspace(0, 1.10 * 0.05, ramp_len)
    idx = _fold_index(n)
    prices = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(prices.double(), window).float()
    bb = rolling_bollinger_percent_b(prices.double(), window).float()

    combo = Combo(lookback_min=window, entry_z=1.5, exit_rule="max_hold",
                  session="24h", signal_family="zscore_only")
    pnl, _trades = run_fold(prices=prices, z_by_window={window: z}, bb_by_window={window: bb},
                    fold_index=idx, combos=[combo], pairs=["TEST"],
                    cost_per_side=torch.tensor([0.0]), device=torch.device("cpu"))
    # A short entered near the jump, held for exactly max_hold_bars (80),
    # forced-closed at a higher price than entry -> a loss (price never came back).
    assert pnl.item() < 0, "forced exit into a non-reverted spike should show the loss, not hide it"


def test_trade_records_carry_entry_side_attribution():
    """Exit-only trade records made B01/B02 un-diagnosable after the fact.

    A spike-then-revert fires a SHORT, so the record must say side=-1, must
    point at an entry_bar strictly before the exit bar, must report a positive
    holding period consistent with those two bars, and must carry the entry
    z-score that actually breached the threshold.
    """
    window = 20
    n = 200
    prices_np = _spike_and_revert_prices(n=n, window=window)
    idx = _fold_index(n)
    prices = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(prices.double(), window).float()
    bb = rolling_bollinger_percent_b(prices.double(), window).float()
    combo = Combo(lookback_min=window, entry_z=1.5, exit_rule="inner_band",
                  session="24h", signal_family="zscore_only")

    _pnl, trades = run_fold(
        prices=prices, z_by_window={window: z}, bb_by_window={window: bb},
        fold_index=idx, combos=[combo], pairs=["TEST"],
        cost_per_side=torch.tensor([0.0]), device=torch.device("cpu"),
    )

    assert trades, "expected at least one trade"
    for t in trades:
        assert t["side"] == -1, "a spike should be faded SHORT"
        assert 0 <= t["entry_bar"] < t["bar"], "entry must precede exit"
        assert t["bars_held"] == t["bar"] - t["entry_bar"]
        assert t["entry_z"] >= combo.entry_z, \
            "entry_z must be the breaching z, not a post-exit value"


def test_trade_records_carry_entry_ts_and_long_side():
    """entry_bar is fold-local, so entry_ts must be stored, not derived.

    Also covers the LONG path, which the short-side test above cannot: a dip
    below -entry_z must record side=+1 and a NEGATIVE entry_z.
    """
    window = 20
    n = 200
    # Mirror image of the spike fixture: a DIP that reverts -> fades LONG.
    base = 1.10
    prices_np = np.full(n, base, dtype=np.float64)
    dip = window + 10
    prices_np[dip:dip + 3] = base * 0.98
    idx = _fold_index(n)
    prices = torch.tensor(prices_np, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(prices.double(), window).float()
    bb = rolling_bollinger_percent_b(prices.double(), window).float()
    combo = Combo(lookback_min=window, entry_z=1.5, exit_rule="inner_band",
                  session="24h", signal_family="zscore_only")

    _pnl, trades = run_fold(
        prices=prices, z_by_window={window: z}, bb_by_window={window: bb},
        fold_index=idx, combos=[combo], pairs=["TEST"],
        cost_per_side=torch.tensor([0.0]), device=torch.device("cpu"),
    )

    assert trades, "expected at least one trade"
    for t in trades:
        assert t["side"] == 1, "a dip should be faded LONG"
        assert t["entry_z"] <= -combo.entry_z
        # The whole reason entry_ts is stored: it must equal the fold index at
        # entry_bar, which a caller holding only the record cannot reconstruct.
        assert t["entry_ts"] == idx[t["entry_bar"]]
        assert t["entry_ts"] < t["ts"]
