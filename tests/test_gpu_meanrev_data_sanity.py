"""Price-sanitation guards.

Regression tests for real corruption found in elthariel/histdata_fx_1m on
2026-08-05, which had silently produced a profit factor of 111.6 for 2005:
  * EURUSD 2001-09-12 close = -0.0001 (negative FX price)
  * AUDJPY 2005 prints near 0.67 against an ~85 median

Both are FINITE numbers, so every NaN-based guard passed them through.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_meanrev.data.loader import OUTLIER_FACTOR, sanitize_closes


def _series(vals, start="2005-01-03"):
    idx = pd.date_range(start, periods=len(vals), freq="min", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


def test_negative_price_is_dropped():
    vals = [1.10] * 100
    vals[50] = -0.0001          # the real EURUSD defect
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_nonpositive"] == 1
    assert (s > 0).all()
    assert len(s) == 99


def test_zero_price_is_dropped():
    vals = [1.10] * 50
    vals[10] = 0.0
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_nonpositive"] == 1
    assert 0.0 not in set(s.values)


def test_off_scale_print_is_dropped():
    """AUDJPY-style: a 0.67 print inside a series trading near 85.

    Asserts the bar is REMOVED, not which of the two stages removed it — a
    grossly off-scale isolated print trips the global band first, and pinning
    the specific bucket would make the test fail on a correct refactor.
    """
    vals = [85.0] * 200
    vals[120] = 0.6742
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_out_of_band"] + rep["dropped_off_scale"] == 1
    assert s.min() > 80


def test_isolated_spike_inside_global_band_is_caught_locally():
    """Stage 2 earns its keep on a series that has DRIFTED a long way.

    On a flat series the two stages overlap (both use a 4x factor), so the
    global band catches everything first. The local filter matters when the
    price level has moved: here the series runs 20 -> 85, so a print of 100
    sits comfortably inside the global band (median ~52, band ~[13, 210]) yet
    is ~5x the local level of ~20 where it occurs.
    """
    vals = list(np.linspace(20.0, 85.0, 3000))
    vals[100] = 100.0                     # ~5x the local level, inside global band
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_out_of_band"] == 0, "should not be caught globally"
    assert rep["dropped_off_scale"] == 1, "...but must be caught locally"
    assert 100.0 not in set(s.values)


def test_real_crash_move_is_preserved():
    """The 2015 CHF unpeg (~-29% in minutes) must SURVIVE the filter.

    This is the whole reason OUTLIER_FACTOR is loose (4x) rather than a tight
    percentage band -- a filter that scrubs genuine regime breaks would be
    worse than the corruption it removes.
    """
    vals = [1.20] * 300 + [0.85] * 300      # EURCHF, 15 Jan 2015
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_off_scale"] == 0, "a real -29% repricing must not be scrubbed"
    assert rep["dropped_nonpositive"] == 0
    assert len(s) == 600


def test_clean_series_is_untouched():
    rng = np.random.default_rng(0)
    vals = 1.10 + np.cumsum(rng.normal(0, 0.0001, 500))
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_nonpositive"] == 0
    assert rep["dropped_off_scale"] == 0
    assert len(s) == 500


def test_outlier_factor_is_loose_enough_for_regime_breaks():
    """Guard the constant itself: anything tighter than 2x would risk real moves."""
    assert OUTLIER_FACTOR >= 3.0


def test_sustained_wrong_scale_segment_is_dropped():
    """The real AUDJPY 2005 defect: a CONTIGUOUS run at the wrong scale.

    A centred rolling median cannot catch this -- the bad run drags the local
    median with it, so the local ratio is ~1 and every bar looks fine. On the
    real archive this left 7,602 rows printing 0.67-2.28 against an ~85 median
    AFTER the local filter had run. Only a whole-series band catches it.
    """
    vals = [85.0] * 2000 + [0.6742] * 800 + [85.0] * 2000   # 800-minute bad run
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_out_of_band"] == 800
    assert s.min() > 80, "sustained wrong-scale segment survived sanitation"
    assert len(s) == 4000


def test_global_band_does_not_scrub_a_long_real_trend():
    """USDJPY-style multi-year drift (76->160 vs ~110 median) must survive."""
    vals = list(np.linspace(76.0, 160.0, 5000))
    s, rep = sanitize_closes(_series(vals))
    assert rep["dropped_out_of_band"] == 0
    assert rep["dropped_off_scale"] == 0
    assert len(s) == 5000


def test_nonpositive_price_blocks_trading_in_backtest():
    """Second line of defence: run_fold must reject a non-positive bar."""
    torch = pytest.importorskip("torch")
    from gpu_meanrev.backtest.batch_walkforward import run_fold
    from gpu_meanrev.features.rolling import rolling_bollinger_percent_b, rolling_zscore
    from gpu_meanrev.signals.mean_reversion import Combo

    n, window = 200, 20
    prices = np.full(n, 1.10)
    prices[window + 10:window + 13] = 1.10 * 1.02
    prices[window + 13:] = 1.10
    prices[150] = -0.0001                      # corrupt bar mid-series
    t = torch.tensor(prices, dtype=torch.float32).unsqueeze(0)
    z = rolling_zscore(t.double(), window).float()
    bb = rolling_bollinger_percent_b(t.double(), window).float()
    idx = pd.date_range("2020-01-06", periods=n, freq="min", tz="UTC")

    combo = Combo(window, 1.5, "inner_band", "24h", "zscore_only")
    pnl, trades = run_fold(
        prices=t, z_by_window={window: z}, bb_by_window={window: bb}, fold_index=idx,
        combos=[combo], pairs=["TEST"], cost_per_side=torch.tensor([0.0]),
        device=torch.device("cpu"),
    )
    # No trade may be booked on the corrupt bar, and nothing astronomical.
    assert all(abs(tr["pnl"]) < 1.0 for tr in trades), "corrupt bar produced an impossible return"
    assert abs(pnl.item()) < 1.0
