"""Leak guards for the multi-timeframe context layer.

Resample-and-join-back is the easiest way to leak the future into a
backtest. These tests exist to make that impossible to do silently.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gpu_meanrev.features.multiframe import (
    align_to_1m,
    build_mtf_context,
    confluence_score,
    resample_closes,
    trend_direction,
)


def _flat_then_spike(n=2000, base=1.10, spike_at=1500, spike=1.05):
    """Dead flat, then a large step up at `spike_at` that never comes back."""
    s = np.full(n, base)
    s[spike_at:] = base * spike
    idx = pd.date_range("2020-01-06", periods=n, freq="min", tz="UTC")  # a Monday
    return pd.Series(s, index=idx)


def test_future_spike_is_not_visible_before_it_happens():
    """The core leak test: nothing before the spike may reflect the spike."""
    closes = _flat_then_spike()
    mtf = build_mtf_context(closes, timeframes=["15m", "1h", "4h"])

    pre = mtf.iloc[:1500]
    # Before the spike the series is perfectly flat -> every EMA diff is 0 ->
    # trend is 0 or NaN. Any +1 before bar 1500 means the future leaked in.
    for col in pre.columns:
        vals = pre[col].dropna()
        assert not (vals > 0).any(), f"{col} saw the future spike before it happened"


def test_htf_bar_is_withheld_until_fully_closed():
    """A 1m bar must see only the PREVIOUS completed higher-timeframe bar."""
    n = 600
    idx = pd.date_range("2020-01-06 00:00", periods=n, freq="min", tz="UTC")
    closes = pd.Series(np.arange(n, dtype=float), index=idx)

    htf = resample_closes(closes, "1h")
    aligned = align_to_1m(htf, idx)

    # The 1h bar labelled 02:00 covers (01:00, 02:00] and closes at 02:00.
    # At 02:00 we must still be seeing the 01:00 bar, not the 02:00 one.
    val_at_0200 = aligned.loc[pd.Timestamp("2020-01-06 02:00", tz="UTC")]
    bar_0100 = htf.loc[pd.Timestamp("2020-01-06 01:00", tz="UTC")]
    bar_0200 = htf.loc[pd.Timestamp("2020-01-06 02:00", tz="UTC")]
    assert val_at_0200 == bar_0100
    assert val_at_0200 != bar_0200


def test_resample_labels_at_close_not_open():
    n = 300
    idx = pd.date_range("2020-01-06 00:00", periods=n, freq="min", tz="UTC")
    closes = pd.Series(np.arange(n, dtype=float), index=idx)
    htf = resample_closes(closes, "1h")
    # First full hour covers (00:00, 01:00] -> labelled 01:00, value = minute 60.
    assert htf.index[0] == pd.Timestamp("2020-01-06 00:00", tz="UTC") or \
           htf.index[0] == pd.Timestamp("2020-01-06 01:00", tz="UTC")
    assert htf.loc[pd.Timestamp("2020-01-06 01:00", tz="UTC")] == 60.0


def test_trend_direction_signs():
    up = pd.Series(np.arange(100, dtype=float))
    down = pd.Series(np.arange(100, 0, -1, dtype=float))
    assert trend_direction(up).iloc[-1] == 1
    assert trend_direction(down).iloc[-1] == -1


def test_confluence_score_bounds_and_nan_handling():
    idx = pd.date_range("2020-01-06", periods=3, freq="min", tz="UTC")
    mtf = pd.DataFrame({
        "trend_1h": [1.0, -1.0, 1.0],
        "trend_4h": [1.0, -1.0, np.nan],   # NaN must be skipped, not counted as 0
    }, index=idx)
    score = confluence_score(mtf)
    assert score.iloc[0] == pytest.approx(1.0)
    assert score.iloc[1] == pytest.approx(-1.0)
    assert score.iloc[2] == pytest.approx(1.0), "NaN timeframe must be ignored, not diluted to 0.5"
